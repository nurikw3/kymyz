"""Судьба Cu_slag: нестационарность vs шум, и вклад динамики (лагов)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from config import ONLINE, CHARGE_COMPOSITION, PRODUCTS
from data import load, time_split


def model():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         subsample=0.8, colsample_bytree=0.8, verbose=-1)


def add_lags(df, cols, lags=(1, 3, 6)):
    out = df.copy()
    new = {}
    for c in cols:
        for L in lags:
            new[f"{c}_lag{L}"] = df[c].shift(L)
    out = pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)
    return out.dropna().reset_index(drop=True)


def main():
    df = load()
    feats = ONLINE + CHARGE_COMPOSITION

    print("=" * 66)
    print("Cu_slag / Fe_slag: хронологический сплит vs случайный 5-fold CV")
    print("=" * 66)
    train, val = time_split(df, 0.2)
    for t in ["Cu_slag", "Fe_slag", "Cu_matte"]:
        m = model(); m.fit(train[feats], train[t])
        r_time = r2_score(val[t], m.predict(val[feats]))
        cv = cross_val_score(model(), df[feats], df[t],
                             cv=KFold(5, shuffle=True, random_state=0),
                             scoring="r2")
        if r_time > 0.5:
            tag = "стационарен, сильный сигнал"
        elif cv.mean() - r_time > 0.3:
            tag = "НЕСТАЦИОНАРНОСТЬ (связь плывёт)"
        else:
            tag = "шум/слабый сигнал"
        print(f"{t:>10} | time-split R2={r_time:>7.3f} | random-CV R2={cv.mean():>6.3f}  -> {tag}")

    print("\n" + "=" * 66)
    print("Вклад динамики: онлайн+состав  vs  +лаги(1,3,6)  [хронологич. сплит]")
    print("=" * 66)
    dfl = add_lags(df, ONLINE, lags=(1, 3, 6))
    lag_feats = feats + [c for c in dfl.columns if "_lag" in c]
    tr, va = time_split(dfl, 0.2)
    for t in ["Cu_slag", "Zn_slag", "CaO_slag", "SiO2_slag", "Fe_slag"]:
        m1 = model(); m1.fit(tr[feats], tr[t]); r1 = r2_score(va[t], m1.predict(va[feats]))
        m2 = model(); m2.fit(tr[lag_feats], tr[t]); r2 = r2_score(va[t], m2.predict(va[lag_feats]))
        print(f"{t:>10} | без лагов R2={r1:>7.3f} | с лагами R2={r2:>7.3f}")


if __name__ == "__main__":
    main()
