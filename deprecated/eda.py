"""EDA + проверка решаемости обратной задачи (онлайн-сигналы -> состав шихты).

Ключевой вопрос: заложил ли генератор синтетики зависимость сигналы = f(состав)?
Если да — soft-sensor шихты (слой 1) реалистичен. Проверяем это baseline-моделью
LightGBM на честном хронологическом сплите и смотрим R2 по каждому таргету.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from config import ONLINE, CHARGE_COMPOSITION, PRODUCTS
from data import load, time_split


def const_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if df[c].nunique() <= 1]


def eval_targets(train, val, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    Xtr, Xva = train[features], val[features]
    for t in targets:
        ytr, yva = train[t], val[t]
        if ytr.nunique() <= 1:
            rows.append({"target": t, "R2": np.nan, "MAE": 0.0,
                         "target_std": float(yva.std()), "note": "константа"})
            continue
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        rows.append({
            "target": t,
            "R2": round(r2_score(yva, pred), 3),
            "MAE": round(mean_absolute_error(yva, pred), 4),
            "target_std": round(float(yva.std()), 4),
            "note": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print(f"Строк: {len(df)}, колонок: {df.shape[1]}")
    print(f"Период: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # Ищем колонки-константы (в синтетике часть состава может быть заморожена).
    consts = const_columns(df, ONLINE + CHARGE_COMPOSITION + PRODUCTS)
    print(f"\nКолонки-константы ({len(consts)}): {consts}")

    train, val = time_split(df, val_frac=0.2)
    print(f"\nTrain: {len(train)}  Val: {len(val)} (хронологический сплит)")

    print("\n" + "=" * 62)
    print("СЛОЙ 1 — обратная задача: онлайн-сигналы -> СОСТАВ ШИХТЫ")
    print("=" * 62)
    r1 = eval_targets(train, val, ONLINE, CHARGE_COMPOSITION)
    print(r1.to_string(index=False))

    print("\n" + "=" * 62)
    print("СЛОЙ 2 — онлайн-сигналы -> ПРОДУКТЫ (штейн/шлак)")
    print("=" * 62)
    r2 = eval_targets(train, val, ONLINE, PRODUCTS)
    print(r2.to_string(index=False))

    print("\nИтог: R2>0.5 означает, что таргет предсказуем по онлайн-сигналам")
    print("(обратная задача решаема). R2<=0 — таргет фактически шум/константа.")


if __name__ == "__main__":
    main()
