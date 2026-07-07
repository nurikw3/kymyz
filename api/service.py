"""Сервисный слой: загрузка .pkl-моделей и обёртки над каждым слоем системы.

Держит один экземпляр пайплайна и детекторов. Детектор смены штабеля — stateful
(буфер потока), поэтому предназначен для одного клиента-симуляции; есть reset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

# делаем модули из src/ импортируемыми (нужно для распаковки .pkl и утилит)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ONLINE, LEVERS, ONLINE_STATE, CHARGE_COMPOSITION  # noqa: E402
from optimize import optimize, constraint_penalty                     # noqa: E402

MODELS = ROOT / "models"


class FurnaceService:
    def __init__(self, models_dir: Path = MODELS):
        self.pipe = joblib.load(models_dir / "pipeline.pkl")
        self.anomaly = joblib.load(models_dir / "anomaly_detector.pkl")
        self.shift = joblib.load(models_dir / "shift_detector.pkl")
        self.df = self.pipe.ref               # исторические данные (для bounds и /sample)

    # ---------- отдельные слои ----------

    def estimate_matte(self, online: dict) -> dict:
        row = pd.DataFrame([online])[ONLINE]
        return {"Cu_matte": round(float(self.pipe.matte_sensor.predict(row).iloc[0]["Cu_matte"]), 3)}

    def estimate_charge(self, online: dict, last_lab: dict | None) -> dict:
        row = pd.DataFrame([online])[ONLINE]
        soft = self.pipe.estimate_charge(row)
        used = dict(last_lab) if last_lab else soft
        return {
            "charge": {k: round(v, 3) for k, v in used.items()},
            "source": "lab_persistence" if last_lab else "soft_sensor",
            "soft_S_charge": round(soft["S_charge"], 3),   # онлайн-оценка серы (кросс-чек)
            "note": "серу видно онлайн; остальное точнее вести последним лаб-замером",
        }

    def twin_predict(self, levers: dict, charge: dict) -> dict:
        out = self.pipe.twin.predict_one({**levers, **charge})
        return {k: round(v, 3) for k, v in out.items()}

    def optimize_regime(self, online: dict, last_lab: dict | None, n_trials: int = 200) -> dict:
        soft = self.pipe.estimate_charge(pd.DataFrame([online])[ONLINE])
        charge = dict(last_lab) if last_lab else soft
        point = {**online, **charge}
        res = optimize(self.pipe.twin, self.df, point, n_trials=n_trials)
        _, viol = constraint_penalty(res["out_after"])
        cu_b, cu_a = res["out_before"]["Cu_slag"], res["out_after"]["Cu_slag"]
        return {
            "recommended_levers": {k: round(res["best_levers"][k], 3) for k in LEVERS},
            "current_levers": {k: round(res["current_levers"][k], 3) for k in LEVERS},
            "cu_slag_before": round(cu_b, 4),
            "cu_slag_after": round(cu_a, 4),
            "cu_slag_gain_pct": round((cu_a - cu_b) / cu_b * 100, 1),
            "predicted_state_after": {k: round(v, 3) for k, v in res["out_after"].items()},
            "feasible": not viol,
            "violations": viol,
        }

    def detect_anomaly(self, online: dict) -> dict:
        return self.anomaly.step(online)

    # ---------- потоковый режим (для симуляции датчиков) ----------

    def stream_step(self, online: dict, last_lab: dict | None, do_optimize: bool = True) -> dict:
        matte = self.estimate_matte(online)
        charge = self.estimate_charge(online, last_lab)
        anomaly = self.anomaly.step(online)
        shift = self.shift.step(online)          # stateful
        result = {
            "matte": matte,
            "charge": charge,
            "regime_anomaly": anomaly,
            "charge_shift": shift,
        }
        if do_optimize:
            result["recommendation"] = self.optimize_regime(online, last_lab, n_trials=150)
        return result

    def reset_stream(self) -> dict:
        self.shift.reset()
        return {"status": "stream buffer reset"}

    # ---------- вспомогательное ----------

    def schema(self) -> dict:
        return {
            "online_inputs": ONLINE,
            "levers": LEVERS,
            "online_state": ONLINE_STATE,
            "charge_composition": CHARGE_COMPOSITION,
            "observable_online": {"strong": ["S_charge"], "partial": ["Fe_charge"],
                                  "weak": ["Cu_charge", "Zn_charge", "CaO_charge", "SiO2_charge"]},
        }

    def sample(self, n: int = 1, tail: bool = True) -> list[dict]:
        """Реальные онлайн-срезы из датасета — для проигрывания в симуляции датчиков."""
        rows = self.df.iloc[-n:] if tail else self.df.iloc[:n]
        return [{k: round(float(r[k]), 4) for k in ONLINE} for _, r in rows.iterrows()]

    def ranges(self) -> dict:
        """Диапазоны (min/p50/max) по онлайн-входам — для генератора-симулятора."""
        out = {}
        for c in ONLINE:
            s = self.df[c]
            out[c] = {"min": round(float(s.min()), 3), "median": round(float(s.median()), 3),
                      "max": round(float(s.max()), 3)}
        return out

    def demo_scenario(self, n: int = 300, n_faults: int = 3) -> dict:
        """Демо-сценарий: реальные срезы с ЗАЛОЖЕННЫМИ отклонениями + разметка.

        Для показа детектора аномалий: наш датасет чистый (сбоев нет), поэтому вставляем
        физичные отклонения (дрейф температуры, скачок SO2, всплеск темп. газов).
        Проигрывать через /stream/step — детектор загорится на размеченных участках.
        """
        import numpy as np
        n = max(n, 90)                       # минимум, чтобы уместить отклонения
        rows = self.df.iloc[-n:].reset_index(drop=True).copy()
        rng = np.random.default_rng(0)
        lo, hi = max(10, n // 10), n - max(60, n // 6)   # валидное окно расстановки
        span = np.arange(lo, hi)
        n_faults = max(1, min(n_faults, len(span)))
        starts = sorted(int(x) for x in rng.choice(span, size=n_faults, replace=False))
        injected = []
        for i, s in enumerate(starts):
            kind = i % 3
            if kind == 0:
                L = 40; rows.loc[s:s + L, "melt_temperature"] += np.linspace(0, 18, L + 1)
                ch, t = "melt_temperature", "дрейф температуры расплава"
            elif kind == 1:
                L = 30; rows.loc[s:s + L, "SO2_out"] += 3.0
                ch, t = "SO2_out", "скачок SO2"
            else:
                L = 25; rows.loc[s:s + L, "offgas_temperature"] += np.linspace(0, 40, L + 1)
                ch, t = "offgas_temperature", "всплеск температуры отходящих газов"
            injected.append({"start": s, "end": s + L, "channel": ch, "type": t})
        samples = [{k: round(float(r[k]), 4) for k in ONLINE} for _, r in rows.iterrows()]
        return {"samples": samples, "injected": injected,
                "note": "проигрывать по порядку через /stream/step; "
                        "regime_anomaly.anomaly загорится на участках injected"}
