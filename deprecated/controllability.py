"""Управляемость выходов рычагами — что вообще можно оптимизировать.

Убираем медленный дрейф (rolling mean) и смотрим, объясняются ли БЫСТРЫЕ отклонения
выхода управляющими рычагами. Если да — выход локально управляем, его можно ставить
в целевую функцию/ограничения. Если остаток = шум — оптимизировать его бессмысленно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from config import LEVERS
from data import load

WIN = 48  # окно детренда ≈ смена


def model():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=30, reg_lambda=1.0, verbose=-1)


def main():
    df = load()
    outputs = ["Cu_slag", "Cu_matte", "melt_temperature", "SO2_out",
               "melt_level", "Fe_slag", "SiO2_slag"]

    print("Доля БЫСТРЫХ отклонений выхода, объяснимая рычагами (детренд, random-CV R2)")
    print("R2>0.3 -> локально управляем рычагами; ~0 -> дрейф/шум, крутить бесполезно")
    print("-" * 70)
    X = df[LEVERS]
    for out in outputs:
        # быстрая компонента = выход минус его скользящее среднее
        trend = df[out].rolling(WIN, center=True, min_periods=1).mean()
        resid = df[out] - trend
        cv = cross_val_score(model(), X, resid,
                             cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
        # и абсолют для сравнения
        cv_abs = cross_val_score(model(), X, df[out],
                                 cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
        tag = "УПРАВЛЯЕМ" if cv.mean() > 0.3 else ("частично" if cv.mean() > 0.1 else "нет")
        print(f"{out:>16} | быстрый R2={cv.mean():>6.3f} | абсолют R2={cv_abs.mean():>6.3f}"
              f" | {tag}")

    # Какие именно рычаги двигают Cu_slag (по importance на детренде).
    print("\nВлияние рычагов на быстрые отклонения Cu_slag (LGBM importance, %):")
    trend = df["Cu_slag"].rolling(WIN, center=True, min_periods=1).mean()
    m = model().fit(X, df["Cu_slag"] - trend)
    imp = pd.Series(m.feature_importances_, index=LEVERS)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print(imp.round(1).to_string())


if __name__ == "__main__":
    main()
