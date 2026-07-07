"""Решающая диагностика структуры данных — определяет всю архитектуру.

Гипотезы:
  A) Продукты (особенно Cu_slag) определяются СОСТАВОМ шихты, а не только
     онлайн-сигналами. Проверка: добавить состав в фичи и посмотреть прирост R2.
  B) Состав шихты кусочно-постоянен ("когда штабель идёт — плюс-минус").
     Проверка: persistence-baseline (значение = лаг) и автокорреляция.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from config import ONLINE, CHARGE_COMPOSITION, PRODUCTS
from data import load, time_split


def fit_r2(train, val, feats, target):
    if train[target].nunique() <= 1:
        return np.nan
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, verbose=-1)
    m.fit(train[feats], train[target])
    return round(r2_score(val[target], m.predict(val[feats])), 3)


def main():
    df = load()
    train, val = time_split(df, 0.2)

    # --- A) Продукты: онлайн vs онлайн+состав шихты ---
    print("=" * 70)
    print("A) Прирост предсказуемости продуктов при добавлении состава шихты")
    print("=" * 70)
    print(f"{'target':>10} | {'онлайн':>8} | {'онлайн+состав':>14}")
    for t in PRODUCTS:
        r_on = fit_r2(train, val, ONLINE, t)
        r_full = fit_r2(train, val, ONLINE + CHARGE_COMPOSITION, t)
        print(f"{t:>10} | {r_on:>8} | {r_full:>14}")

    # --- B) Персистентность состава шихты (кусочно-постоянство) ---
    print("\n" + "=" * 70)
    print("B) Насколько состав шихты 'держится' во времени (шаг = 10 мин)")
    print("=" * 70)
    print(f"{'target':>11} | {'autocorr@1':>10} | {'autocorr@6(1ч)':>14} | "
          f"{'autocorr@144(1сут)':>18}")
    for t in CHARGE_COMPOSITION:
        s = df[t]
        a1 = round(s.autocorr(1), 3)
        a6 = round(s.autocorr(6), 3)
        a144 = round(s.autocorr(144), 3)
        print(f"{t:>11} | {a1:>10} | {a6:>14} | {a144:>18}")

    # persistence baseline: y[t] = y[t-1] (насколько мал шаг изменения)
    print("\nPersistence-baseline для состава (насколько точно 'вчера=сегодня'):")
    print(f"{'target':>11} | {'MAE persist':>12} | {'std':>8} | {'|Δ| median/step':>16}")
    for t in CHARGE_COMPOSITION:
        s = df[t]
        mae_p = mean_absolute_error(s.iloc[1:], s.shift(1).iloc[1:])
        dstep = s.diff().abs().median()
        print(f"{t:>11} | {mae_p:>12.4f} | {s.std():>8.3f} | {dstep:>16.5f}")

    # --- Топ-корреляции с Cu_slag (что вообще с ним связано) ---
    print("\n" + "=" * 70)
    print("Топ-12 |корреляций| с Cu_slag (линейная связь)")
    print("=" * 70)
    num = df.drop(columns=["timestamp"])
    corr = num.corr()["Cu_slag"].drop("Cu_slag").abs().sort_values(ascending=False)
    print(corr.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
