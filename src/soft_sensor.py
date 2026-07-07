"""Виртуальный анализатор штейна (soft-sensor Cu_matte).

Боль: содержание Cu в штейне оператор узнаёт из лаборатории раз в смену. Между
замерами работает «на глаз». Soft-sensor оценивает Cu_matte по онлайн-сигналам печи
каждые 10 минут — закрывает информационный лаг.

Вход — ТОЛЬКО онлайн-сигналы (без лабораторных величин). Cu_matte стационарен
(time-split R²≈0.96), поэтому честно предсказуем вперёд.

Демо-метрика: на точках МЕЖДУ лабораторными замерами сравниваем ошибку
  - persistence (как сейчас: держим последний замер) vs
  - soft-sensor (наша оценка)
относительно истинного значения.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import ONLINE
from data import load, time_split

SOFT_TARGETS = ["Cu_matte"]   # стационарные лаб-величины, предсказуемые из онлайн
LAB_INTERVAL = 48             # лаборатория даёт замер раз в смену (48*10мин = 8ч)
MODELS_DIR = Path("models")
REPORTS = Path("reports")


def _model() -> LGBMRegressor:
    return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                         min_child_samples=30, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


class SoftSensor:
    def __init__(self, targets=SOFT_TARGETS, features=ONLINE):
        self.targets = list(targets)
        self.features = list(features)
        self.models: dict[str, LGBMRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "SoftSensor":
        for t in self.targets:
            self.models[t] = _model().fit(df[self.features], df[t])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: m.predict(df[self.features])
                             for t, m in self.models.items()}, index=df.index)

    def save(self, path: Path = MODELS_DIR) -> None:
        import joblib
        path.mkdir(exist_ok=True)
        joblib.dump(self, path / "soft_sensor.pkl")


def rolling_quality(df: pd.DataFrame, target: str, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s, maes = [], []
    for tr, va in tscv.split(df):
        m = _model().fit(df.iloc[tr][ONLINE], df.iloc[tr][target])
        pred = m.predict(df.iloc[va][ONLINE])
        r2s.append(r2_score(df.iloc[va][target], pred))
        maes.append(mean_absolute_error(df.iloc[va][target], pred))
    return {"R2_mean": round(float(np.mean(r2s)), 3),
            "R2_last": round(float(r2s[-1]), 3),
            "MAE_mean": round(float(np.mean(maes)), 4)}


def demo_vs_persistence(df: pd.DataFrame, sensor: SoftSensor, target: str) -> dict:
    """На валидации: сравнить soft-sensor и persistence на точках МЕЖДУ замерами."""
    _, va = time_split(df, 0.2)
    va = va.reset_index(drop=True)
    truth = va[target].to_numpy()
    pred = sensor.predict(va)[target].to_numpy()

    # persistence: последний лаб-замер, обновляется каждые LAB_INTERVAL шагов
    last_lab = np.empty_like(truth)
    for i in range(len(truth)):
        anchor = (i // LAB_INTERVAL) * LAB_INTERVAL
        last_lab[i] = truth[anchor]

    # оцениваем только точки МЕЖДУ замерами (где оператор реально «слепой»)
    between = np.array([i % LAB_INTERVAL != 0 for i in range(len(truth))])
    mae_soft = mean_absolute_error(truth[between], pred[between])
    mae_pers = mean_absolute_error(truth[between], last_lab[between])
    return {
        "target": target,
        "MAE_persistence": round(float(mae_pers), 4),
        "MAE_softsensor": round(float(mae_soft), 4),
        "improvement_pct": round((1 - mae_soft / mae_pers) * 100, 1),
    }


def save_plot(df: pd.DataFrame, sensor: SoftSensor, target: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, va = time_split(df, 0.2)
    va = va.reset_index(drop=True).iloc[:300]     # первые ~2 суток валидации
    truth = va[target].to_numpy()
    pred = sensor.predict(va)[target].to_numpy()
    lab_idx = np.arange(0, len(va), LAB_INTERVAL)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(truth, color="#888", lw=1, label="истина (непрерывно)")
    ax.plot(pred, color="#1f77b4", lw=1.6, label="soft-sensor (каждые 10 мин)")
    ax.scatter(lab_idx, truth[lab_idx], color="#d62728", zorder=5, s=40,
               label="лаб-замер (раз в смену)")
    ax.step(np.arange(len(va)),
            truth[(np.arange(len(va)) // LAB_INTERVAL) * LAB_INTERVAL],
            color="#d62728", ls="--", lw=1, alpha=0.6, where="post",
            label="сейчас у оператора (последний замер)")
    ax.set_title(f"Виртуальный анализатор штейна: {target}")
    ax.set_xlabel("шаг (10 мин)"); ax.set_ylabel(f"{target}, %")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / f"softsensor_{target}.png", dpi=120)
    plt.close(fig)


def main():
    df = load()
    results = {}
    for t in SOFT_TARGETS:
        q = rolling_quality(df, t)
        print(f"[{t}] rolling-CV: R2_mean={q['R2_mean']}  R2_last={q['R2_last']}  "
              f"MAE_mean={q['MAE_mean']}")
        results[t] = {"rolling": q}

    # финальный сенсор — на актуальном окне (как двойник)
    sensor = SoftSensor().fit(df.iloc[-2016:])
    sensor.save()

    print("\nДемо: точность МЕЖДУ лабораторными замерами (раз в смену)")
    print("-" * 60)
    for t in SOFT_TARGETS:
        d = demo_vs_persistence(df, sensor, t)
        print(f"[{t}] MAE persistence(как сейчас)={d['MAE_persistence']}  "
              f"soft-sensor={d['MAE_softsensor']}  -> точнее на {d['improvement_pct']}%")
        results[t]["demo"] = d
        save_plot(df, sensor, t)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "soft_sensor_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nСохранено: models/soft_sensor.pkl, reports/soft_sensor_result.json,"
          " reports/figures/softsensor_*.png")


if __name__ == "__main__":
    main()
