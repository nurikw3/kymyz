"""Обучение всех моделей и сохранение артефактов в models/*.pkl.

Единая точка сборки: обучает сквозной пайплайн (оценка шихты, штейн, двойник) и оба
онлайн-детектора, сохраняет их в .pkl для загрузки API-сервисом.

Запуск:  PYTHONPATH=src .venv/bin/python src/train.py
"""
from __future__ import annotations

from pathlib import Path

import joblib

from data import load
from pipeline import FurnacePipeline
from detectors import RegimeAnomalyDetector, ChargeShiftDetector

MODELS = Path("models")


def main() -> None:
    df = load()
    print(f"Данные: {len(df)} строк")

    print("Обучение сквозного пайплайна (шихта + штейн + двойник)...")
    pipe = FurnacePipeline().fit(df)

    print("Обучение детектора отклонений режима...")
    anomaly = RegimeAnomalyDetector().fit(df)

    print("Обучение детектора смены штабеля...")
    shift = ChargeShiftDetector().fit(df)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS / "pipeline.pkl")
    joblib.dump(anomaly, MODELS / "anomaly_detector.pkl")
    joblib.dump(shift, MODELS / "shift_detector.pkl")

    # чистим устаревшие .joblib
    for old in MODELS.glob("*.joblib"):
        old.unlink()

    print("\nСохранено:")
    for p in sorted(MODELS.glob("*.pkl")):
        print(f"  {p}  ({p.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
