"""Загрузка датасета и корректный по времени train/val сплит (без перемешивания)."""
from __future__ import annotations

import pandas as pd

from config import DATA_PATH, TIMESTAMP


def load(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP])
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, val_frac: float = 0.2):
    """Хронологический сплит: последние val_frac идут в валидацию (без утечки)."""
    n = len(df)
    cut = int(n * (1 - val_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
