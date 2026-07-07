"""Цифровой двойник печи Ванюкова.

Двойник моделирует отклик печи на управление:
    f(рычаги оператора, состав шихты) -> продукты + состояние + экология

Используется оптимизатором (optimize.py): при фиксированном (оценённом) составе
шихты крутим рычаги и предсказываем Cu_slag / Cu_matte / температуру / SO2 / уровень.

Нестационарность Cu_slag/Fe_slag решается двумя приёмами:
  1. Честная оценка через rolling-origin CV (TimeSeriesSplit) — не обманываем себя.
  2. Финальный двойник обучается на ПОСЛЕДНЕМ окне (recent-window), т.к. он должен
     отражать ТЕКУЩИЙ режим печи, а не усреднять уплывшую историю.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import LEVERS, CHARGE_COMPOSITION
from data import load

# Вход двойника: то, чем управляем + контекст шихты.
TWIN_FEATURES = LEVERS + CHARGE_COMPOSITION

# Выход: что хотим предсказать (для целевой функции и ограничений).
TWIN_OUTPUTS = [
    "Cu_slag",          # потери меди — минимизируем
    "Cu_matte",         # кондиция штейна — ограничение
    "melt_temperature", # тепловой режим — ограничение
    "SO2_out",          # экология — ограничение
    "melt_level",       # уровень ванны — ограничение
    "Fe_slag",
    "SiO2_slag",
    "Zn_slag",
    "CaO_slag",
]

MODELS_DIR = Path("models")


def _make_model() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,          # немного регуляризации против переобучения
        verbose=-1,
    )


class DigitalTwin:
    """Multi-output обёртка: по одной LGBM-модели на каждый выход."""

    def __init__(self, features=TWIN_FEATURES, outputs=TWIN_OUTPUTS):
        self.features = list(features)
        self.outputs = list(outputs)
        self.models: dict[str, LGBMRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "DigitalTwin":
        X = df[self.features]
        for out in self.outputs:
            m = _make_model()
            m.fit(X, df[out])
            self.models[out] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features]
        return pd.DataFrame(
            {out: m.predict(X) for out, m in self.models.items()},
            index=X.index,
        )

    def predict_one(self, row: dict) -> dict:
        """Предсказание для одной точки (dict рычагов+состава) — для оптимизатора."""
        X = pd.DataFrame([row])[self.features]
        return {out: float(m.predict(X)[0]) for out, m in self.models.items()}

    def save(self, path: Path = MODELS_DIR) -> None:
        import joblib
        path.mkdir(exist_ok=True)
        joblib.dump(self, path / "twin.pkl")

    @staticmethod
    def load(path: Path = MODELS_DIR) -> "DigitalTwin":
        import joblib
        return joblib.load(path / "twin.pkl")


def rolling_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Rolling-origin валидация: обучаемся на прошлом, тестируем на следующем окне.

    Честно моделирует эксплуатацию (обучил на недавней истории -> предсказал вперёд)
    и корректно наказывает нестационарность.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for out in TWIN_OUTPUTS:
        r2s, maes = [], []
        for tr_idx, va_idx in tscv.split(df):
            tr, va = df.iloc[tr_idx], df.iloc[va_idx]
            m = _make_model()
            m.fit(tr[TWIN_FEATURES], tr[out])
            pred = m.predict(va[TWIN_FEATURES])
            r2s.append(r2_score(va[out], pred))
            maes.append(mean_absolute_error(va[out], pred))
        rows.append({
            "output": out,
            "R2_mean": round(float(np.mean(r2s)), 3),
            "R2_last": round(float(r2s[-1]), 3),   # ближайшее к "сейчас" окно
            "MAE_mean": round(float(np.mean(maes)), 4),
        })
    return pd.DataFrame(rows)


def build(recent_window: int | None = None, save: bool = True) -> DigitalTwin:
    """Финальный двойник. recent_window — обучать только на последних N строках
    (для нестационарного режима); None — на всех данных."""
    df = load()
    train_df = df if recent_window is None else df.iloc[-recent_window:]
    twin = DigitalTwin().fit(train_df)
    if save:
        twin.save()
    return twin


def main() -> None:
    df = load()
    print("Rolling-origin CV двойника (обучаюсь на прошлом -> предсказываю вперёд):\n")
    cv = rolling_cv(df, n_splits=5)
    print(cv.to_string(index=False))
    print("\nR2_last = качество на ПОСЛЕДНЕМ окне (ближе всего к реальной эксплуатации).")

    # Сравним: двойник на всех данных vs только на последнем окне (recent-window).
    print("\n" + "=" * 60)
    print("Финальный двойник: обучаем на последнем окне (~2 недели = 2016 шагов)")
    print("=" * 60)
    twin = build(recent_window=2016, save=True)
    print(f"Обучен на {2016} последних точках. Выходы: {twin.outputs}")
    print("Сохранён в models/twin.pkl")

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "twin_cv.json").write_text(cv.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
