"""Онлайн-детекторы для потоковой работы (один срез датчиков за раз).

В отличие от батч-версий в reports/анализе, эти классы предобучаются и затем работают
инкрементально — принимают текущий срез и держат минимальное состояние. Нужны для
API/симуляции, где датчики отдают данные потоком.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import LEVERS, ONLINE, ONLINE_STATE
from soft_sensor import SoftSensor

MONITOR = ["SO2_out", "melt_temperature", "melt_level", "offgas_temperature"]
Z_ANOMALY = 5.0        # порог отклонения режима (в референс-сигмах)
SHIFT_LAG = 6          # окно сравнения оценки состава (60 мин)
SHIFT_DISTANCE = 24    # не флагать смену чаще, чем раз в 4 ч


def _features_for(y: str) -> list[str]:
    return LEVERS + [c for c in ONLINE_STATE if c != y]


def _model() -> LGBMRegressor:
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=30, reg_lambda=1.0, verbose=-1)


class RegimeAnomalyDetector:
    """Отклонение режима: наблюдаемая онлайн-величина против эталона (факт vs ожидание)."""

    def __init__(self, z: float = Z_ANOMALY):
        self.z = z
        self.models: dict[str, tuple] = {}

    def fit(self, df: pd.DataFrame) -> "RegimeAnomalyDetector":
        for y in MONITOR:
            m = _model().fit(df[_features_for(y)], df[y])
            sd = float((df[y] - m.predict(df[_features_for(y)])).std())
            self.models[y] = (m, sd)
        return self

    def step(self, snapshot: dict) -> dict:
        row = pd.DataFrame([snapshot])
        worst_ch, worst_z = "", 0.0
        for y, (m, sd) in self.models.items():
            pred = float(m.predict(row[_features_for(y)])[0])
            zc = abs(snapshot[y] - pred) / (sd + 1e-9)
            if zc > worst_z:
                worst_ch, worst_z = y, zc
        return {"anomaly": bool(worst_z > self.z),
                "channel": worst_ch, "z": round(worst_z, 2)}


class ChargeShiftDetector:
    """Смена штабеля: change-point на онлайн-оценке состава шихты soft-sensor'ом."""

    def __init__(self, comp=("S_charge", "Fe_charge")):
        self.comp = list(comp)
        self.sensor: SoftSensor | None = None
        self.ch_std: dict[str, float] = {}
        self.threshold = 1.0
        self.buffer: deque = deque(maxlen=SHIFT_LAG + 1)
        self.last_flag = -SHIFT_DISTANCE
        self.t = 0

    def fit(self, df: pd.DataFrame) -> "ChargeShiftDetector":
        self.sensor = SoftSensor(targets=self.comp, features=ONLINE).fit(df)
        est = self.sensor.predict(df)
        score = np.zeros(len(df))
        for c in self.comp:
            e = est[c].to_numpy()
            ch = np.abs(e - np.concatenate([np.zeros(SHIFT_LAG), e[:-SHIFT_LAG]]))
            ch[:SHIFT_LAG] = 0
            sd = float(np.std(ch)) + 1e-9
            self.ch_std[c] = sd
            score += ch / sd
        score /= len(self.comp)
        self.threshold = float(np.percentile(score, 94))
        return self

    def reset(self) -> None:
        self.buffer.clear()
        self.last_flag = -SHIFT_DISTANCE
        self.t = 0

    def step(self, snapshot: dict) -> dict:
        row = pd.DataFrame([snapshot])[ONLINE]
        est = {c: float(self.sensor.models[c].predict(row)[0]) for c in self.comp}
        self.buffer.append(est)
        self.t += 1
        shift, score = False, 0.0
        if len(self.buffer) > SHIFT_LAG:
            old = self.buffer[0]
            score = sum(abs(est[c] - old[c]) / self.ch_std[c] for c in self.comp) / len(self.comp)
            if score > self.threshold and self.t - self.last_flag > SHIFT_DISTANCE:
                shift = True
                self.last_flag = self.t
        return {"charge_shift": bool(shift), "score": round(float(score), 3),
                "threshold": round(self.threshold, 3),
                "estimated_composition": {c: round(v, 3) for c, v in est.items()}}
