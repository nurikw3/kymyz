"""Как честно предсказать Cu_slag вперёд. Проверяем калибровку по лаб-замеру.

Реальный сценарий: лаборатория даёт Cu_slag периодически (раз в смену). Между
замерами soft-sensor держит последнее значение и корректирует его по онлайн-сигналам.
Формально: добавляем в фичи "последний известный лаб-замер" (лаг на горизонт смены)
и смотрим, спасает ли это нестационарность.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from config import ONLINE, CHARGE_COMPOSITION
from data import load, time_split

SHIFT = 48  # 48 шагов * 10 мин = 8 часов ≈ смена (лаг лабораторного замера)


def r2_mae(y, p):
    return round(r2_score(y, p), 3), round(mean_absolute_error(y, p), 4)


def model():
    return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                         min_child_samples=30, reg_lambda=1.0, verbose=-1)


def main():
    df = load()
    tgt = "Cu_slag"

    # Автокорреляция — насколько Cu_slag "гладкий"/дрейфующий.
    s = df[tgt]
    print(f"Cu_slag autocorr: @1={s.autocorr(1):.3f}  @6(1ч)={s.autocorr(6):.3f}  "
          f"@48(смена)={s.autocorr(SHIFT):.3f}  @144(сут)={s.autocorr(144):.3f}")
    print(f"std={s.std():.4f}\n")

    # Готовим фичи с "последним лаб-замером" (значение 8ч назад).
    d = df.copy()
    d["Cu_slag_lab"] = d[tgt].shift(SHIFT)       # последний доступный замер
    d = d.dropna().reset_index(drop=True)
    tr, va = time_split(d, 0.2)

    print("Хронологический сплит, честная оценка вперёд:")
    print("-" * 60)

    # 1) baseline: persistence — просто держим последний замер.
    r, m = r2_mae(va[tgt], va["Cu_slag_lab"])
    print(f"persistence (последний замер, без модели) : R2={r:>7} MAE={m}")

    # 2) онлайн-сигналы + состав, без замера.
    f2 = ONLINE + CHARGE_COMPOSITION
    mdl = model().fit(tr[f2], tr[tgt])
    r, m = r2_mae(va[tgt], mdl.predict(va[f2]))
    print(f"онлайн+состав (без лаб-замера)            : R2={r:>7} MAE={m}")

    # 3) онлайн + состав + последний лаб-замер (калибровка).
    f3 = f2 + ["Cu_slag_lab"]
    mdl = model().fit(tr[f3], tr[tgt])
    pred3 = mdl.predict(va[f3])
    r, m = r2_mae(va[tgt], pred3)
    print(f"онлайн+состав+лаб-замер (КАЛИБРОВКА)       : R2={r:>7} MAE={m}")

    # 4) residual: предсказываем ОТКЛОНЕНИЕ от последнего замера.
    tr_res = tr[tgt] - tr["Cu_slag_lab"]
    mdl = model().fit(tr[f2], tr_res)
    pred4 = va["Cu_slag_lab"] + mdl.predict(va[f2])
    r, m = r2_mae(va[tgt], pred4)
    print(f"замер + модель отклонения (residual)      : R2={r:>7} MAE={m}")


if __name__ == "__main__":
    main()
