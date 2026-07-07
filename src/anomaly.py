"""Детектор отклонений технологического режима (раннее выявление).

Из ТЗ: «раннее выявление отклонений технологического режима». Механика — эталон
нормальной связи между сигналами: каждую ОНЛАЙН-величину предсказываем по управлению
и остальным онлайн-сигналам (модель обучена на «нормальном» референс-периоде). Пока
печь работает штатно, факт ≈ эталон. Когда режим отклоняется (сбой, нештатное
состояние, дрейф), наблюдаемая величина расходится с ожиданием — это ловим.

Остаток нормируем по СКОЛЬЗЯЩЕМУ окну (отделяем всплески/сдвиги от медленного дрейфа):
  - большой мгновенный |z| -> точечная аномалия;
  - устойчивый сдвиг (CUSUM) -> дрейф режима.

Детектор смены штабеля: состав шихты не виден в мгновенном change-score сырых сигналов
(шум рычагов маскирует скачок), но виден в УРОВНЕ сигналов (S_charge R²≈0.78 через SO2).
Поэтому смену ловим как change-point на ОЦЕНКЕ состава от soft-sensor'а — оценка
следует за истинным составом и её сдвиг = новая партия. Это опережает лабораторию.

Валидация: инъекция искусственных отклонений в held-out часть и замер recall/задержки.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import LEVERS, ONLINE, ONLINE_STATE
from data import load
from soft_sensor import SoftSensor

# Мониторим ОНЛАЙН-величины (доступны в реальном времени):
MONITOR = ["SO2_out", "melt_temperature", "melt_level", "offgas_temperature"]
REF_FRAC = 0.6          # первые 60% — «нормальный» референс
ROLL = 144              # окно нормализации остатка (сутки)
Z_ANOMALY = 5.0         # порог точечной аномалии (в референс-сигмах)
CUSUM_K = 1.0
CUSUM_H = 20.0
LAB_INTERVAL = 48       # лаборатория даёт анализ раз в смену (48*10мин = 8ч)
REPORTS = Path("reports")


def _model():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=30, reg_lambda=1.0, verbose=-1)


def _features_for(y: str) -> list[str]:
    return LEVERS + [c for c in ONLINE_STATE if c != y]


def fit_reference(df: pd.DataFrame) -> dict:
    """Эталон-модели на референс-периоде + std остатка (для нормировки)."""
    n_ref = int(len(df) * REF_FRAC)
    ref = df.iloc[:n_ref]
    models = {}
    for y in MONITOR:
        m = _model().fit(ref[_features_for(y)], ref[y])
        resid = ref[y] - m.predict(ref[_features_for(y)])
        models[y] = (m, float(resid.std()))
    return models


def score_frame(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Остаток эталона, нормированный на референс-разброс (данные стационарны).
    Ловит и ступенчатые, и медленные (ramp) отклонения, в отличие от rolling-нормировки."""
    out = pd.DataFrame(index=df.index)
    for y, (m, sd) in models.items():
        resid = df[y] - m.predict(df[_features_for(y)])
        out[y] = resid / (sd + 1e-9)
    out["score"] = out[MONITOR].abs().max(axis=1)   # max по каналам — любой канал важен
    return out


def cusum(series: np.ndarray, k: float, h: float) -> np.ndarray:
    sp = sm = 0.0
    flags = np.zeros(len(series), dtype=bool)
    for i, x in enumerate(series):
        sp = max(0.0, sp + x - k)
        sm = min(0.0, sm + x + k)
        if sp > h or sm < -h:
            flags[i] = True
            sp = sm = 0.0
    return flags


def inject_faults(df: pd.DataFrame, rng: np.random.Generator):
    """Вставляем искусственные отклонения в held-out для честной валидации детектора."""
    d = df.copy().reset_index(drop=True)
    n_ref = int(len(d) * REF_FRAC)
    span = np.arange(n_ref + ROLL, len(d) - 200)
    starts = np.sort(rng.choice(span, size=6, replace=False))
    episodes = []
    for i, s in enumerate(starts):
        kind = i % 3
        if kind == 0:      # дрейф температуры расплава
            L = 80; d.loc[s:s+L, "melt_temperature"] += np.linspace(0, 18, L + 1)
        elif kind == 1:    # ступенчатый скачок SO2
            L = 60; d.loc[s:s+L, "SO2_out"] += 3.0
        else:              # всплеск температуры отходящих газов
            L = 40; d.loc[s:s+L, "offgas_temperature"] += np.linspace(0, 40, L + 1)
        episodes.append((int(s), int(s + L), kind))
    return d, episodes


def detect_charge_shifts(df: pd.DataFrame):
    """Смена штабеля через change-point на оценке состава soft-sensor'ом.
    Возвращает (детекты, истинные смены, recall, среднее опережение лаборатории, ч)."""
    n_ref = int(len(df) * REF_FRAC)
    comp = ["S_charge", "Fe_charge"]                 # наблюдаемые онлайн компоненты
    sensor = SoftSensor(targets=comp, features=ONLINE).fit(df.iloc[:n_ref])
    est = sensor.predict(df)

    # комбинированный нормированный change-score оценки состава
    score = np.zeros(len(df))
    for c in comp:
        e = est[c].to_numpy()
        ch = np.abs(e - np.concatenate([np.zeros(6), e[:-6]]))
        ch[:6] = 0
        score += ch / (np.std(ch) + 1e-9)
    score /= len(comp)

    # смена штабеля = ПИК change-score оценки (локальный максимум выше порога)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(score, height=np.percentile(score, 94),
                          distance=LAB_INTERVAL // 2)

    # истинные смены штабеля — ступеньки истинного состава
    s = df["S_charge"].to_numpy()
    steps = np.where(np.abs(s - np.concatenate([[s[0]], s[:-1]])) > 0.5)[0]
    true_shifts = [i for k, i in enumerate(steps)
                   if (k == 0 or i - steps[k - 1] > LAB_INTERVAL) and i >= n_ref]
    det_eval = [int(d) for d in peaks if d >= n_ref]

    leads, hits = [], 0
    for ts in true_shifts:
        near = [d for d in det_eval if ts - 6 <= d <= ts + LAB_INTERVAL]
        if near:
            hits += 1
            leads.append((ts + LAB_INTERVAL - min(near)) * 10 / 60.0)  # часов до лаборатории
    recall = hits / len(true_shifts) if true_shifts else float("nan")
    return det_eval, true_shifts, recall, (np.mean(leads) if leads else 0.0)


def evaluate(episodes, anomalies: np.ndarray):
    """Recall эпизодов + задержка детекта (в минутах от начала эпизода)."""
    hits, latencies = 0, []
    for s, e, _ in episodes:
        fired = np.where(anomalies[s:e + 1])[0]
        if len(fired):
            hits += 1
            latencies.append(fired.min() * 10)   # шаг = 10 мин
    recall = hits / len(episodes) if episodes else float("nan")
    return recall, latencies


def main():
    df = load()

    # 1) детекция на реальных данных (без инъекций) — сколько естественных отклонений
    models = fit_reference(df)
    sc = score_frame(df, models)
    n_ref = int(len(df) * REF_FRAC)
    work = np.zeros(len(df), bool); work[n_ref:] = True
    anomalies = (sc["score"] > Z_ANOMALY) & work
    drift = cusum((sc["score"] - np.median(sc["score"])).to_numpy(), CUSUM_K, CUSUM_H) & work

    print("=" * 62)
    print("ДЕТЕКТОР ОТКЛОНЕНИЙ РЕЖИМА")
    print("=" * 62)
    print(f"Естественные точечные аномалии (|z|>{Z_ANOMALY}): {int(anomalies.sum())}")
    print(f"Эпизоды дрейфа режима (CUSUM):                 {int(drift.sum())}")

    # 2) честная валидация: инъекция искусственных отклонений
    rng = np.random.default_rng(0)
    dfi, episodes = inject_faults(df, rng)
    sci = score_frame(dfi, models)
    anom_i = (sci["score"] > Z_ANOMALY).to_numpy()
    recall, lat = evaluate(episodes, anom_i)

    print(f"\nВалидация инъекцией ({len(episodes)} искусственных отклонений):")
    print(f"  поймано (recall):        {recall*100:.0f}%")
    if lat:
        print(f"  средняя задержка детекта: {np.mean(lat):.0f} мин "
              f"(медиана {np.median(lat):.0f} мин)")

    # 3) детектор смены штабеля через soft-sensor состава
    det, true_sh, sh_recall, lead_h = detect_charge_shifts(df)
    print(f"\nДетектор смены штабеля (через оценку состава soft-sensor'ом):")
    print(f"  истинных смен партии:    {len(true_sh)}")
    print(f"  поймано (recall):        {sh_recall*100:.0f}%")
    print(f"  опережение лаборатории:  {lead_h:.1f} ч (лаборатория показала бы состав "
          f"только через смену)")

    save_plot(dfi, sci, anom_i, episodes)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "anomaly_result.json").write_text(json.dumps({
        "monitored": MONITOR,
        "natural_point_anomalies": int(anomalies.sum()),
        "natural_drift_episodes": int(drift.sum()),
        "injection_recall_pct": round(float(recall * 100), 1),
        "injection_latency_min_mean": round(float(np.mean(lat)), 0) if lat else None,
        "charge_shift_true": len(true_sh),
        "charge_shift_recall_pct": round(float(sh_recall * 100), 1),
        "charge_shift_lead_hours": round(float(lead_h), 1),
    }, ensure_ascii=False, indent=2))
    print("\nСохранено: reports/anomaly_result.json, reports/figures/anomaly.png")


def save_plot(df, sc, anomalies, episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ref = int(len(df) * REF_FRAC)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, sc["score"], color="#1f77b4", lw=0.7, label="anomaly score (max по каналам)")
    ax.axhline(Z_ANOMALY, color="#d62728", ls="--", lw=1, label=f"порог ({Z_ANOMALY}σ)")
    for i, (s, e, _) in enumerate(episodes):
        ax.axvspan(s, e, color="#2ca02c", alpha=0.18,
                   label="инъецированное отклонение" if i == 0 else None)
    ax.scatter(x[anomalies], sc["score"].to_numpy()[anomalies], color="#d62728", s=8,
               zorder=5, label="детект")
    ax.axvspan(0, n_ref, color="#eee", alpha=0.5)
    ax.text(n_ref / 2, ax.get_ylim()[1] * 0.9, "референс", ha="center", fontsize=8, color="#888")
    ax.set_title("Детектор отклонений режима: инъецированные отклонения и детекты")
    ax.set_xlabel("шаг (10 мин)"); ax.set_ylabel("score")
    ax.legend(fontsize=8, ncol=2, loc="upper left"); fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "anomaly.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
