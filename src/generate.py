"""Физический генератор синтетических данных печи Ванюкова.

Зачем: исходная синтетика (mics/dataset(in).csv) физически неправдоподобна — состав
шихты не влияет на онлайн-сигналы, Cu_slag нестационарен. Из-за этого soft-sensor
шихты и детектор смены штабеля не работают в принципе. Здесь мы генерируем данные с
КАУЗАЛЬНО ВЕРНОЙ физикой автогенной плавки на штейн:

    состав шихты + режим дутья  ->  штейн / шлак / газы / температуры / котёл

Ключевые заложенные закономерности (источник физики — диссертация в mics/):
  * степень окисления ox растёт с кислородом/обогащением, падает с расходом материала;
  * Cu_matte (матность штейна) растёт с ox — окисляем Fe, штейн богатеет;
  * Cu_slag (потери меди) — U-образная по ox: недокисление -> бедный штейн, перекисление
    -> магнетит и механические потери. Есть ОПТИМУМ -> оптимизатору есть что искать;
  * SO2 и температура несут след состава (S, Cu) -> состав наблюдаем онлайн -> работают
    soft-sensor шихты и детектор смены штабеля;
  * состав кусочно-постоянен (штабели) со ступенчатыми сменами;
  * связи стационарны во времени -> честный прогноз вперёд возможен;
  * инерция ванны (лаги) + измерительный шум.

Выход: data/dataset.csv с теми же 35 колонками, что и оригинал.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N = 13057
FREQ = "10min"
START = "2026-04-08 00:00:00"
SEED = 42

# Базовые уровни (means) — целимся в диапазоны реального процесса.
B = dict(
    Cu_charge=16.0, Fe_charge=27.4, SiO2_charge=12.6, CaO_charge=1.30,
    Zn_charge=3.80, S_charge=32.0, charge_moisture=4.0,
    charge_flow=94.0, concentrate_flow=68.0, coal_flow=7.0, coal_dust_flow=3.3,
    addition_flow=17.0, air_flow=79.0, oxygen_flow=201.0, oxygen_concentration=50.0,
    natural_gas_flow=800.0,
)
# Разброс драйверов (для нормировки в физических формулах).
SD = dict(
    Cu_charge=2.6, Fe_charge=1.6, SiO2_charge=1.2, CaO_charge=0.12, Zn_charge=0.9,
    S_charge=1.6, oxygen_flow=6.0, oxygen_concentration=4.5, air_flow=3.0,
    concentrate_flow=4.0, addition_flow=1.6, natural_gas_flow=60.0, coal_flow=0.5,
)


def z(x, k):
    return (x - B[k]) / SD[k]


def ewm(a: np.ndarray, span: float) -> np.ndarray:
    """Инерция ванны: экспоненциальное сглаживание (вводит физический лаг)."""
    return pd.Series(a).ewm(span=span).mean().to_numpy()


def ar_process(rng, n, sd, phi=0.94, base=0.0):
    """Стационарный AR(1) вокруг base с целевым стационарным std=sd и автокорр.=phi."""
    eps = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return base + x


def piecewise_charge(rng, n, key, seg_min=144, seg_max=432):
    """Кусочно-постоянный состав шихты (штабели) со ступенчатыми сменами партий."""
    vals = np.empty(n)
    i = 0
    while i < n:
        seg = rng.integers(seg_min, seg_max)
        level = rng.normal(B[key], SD.get(key, B[key] * 0.06))
        j = min(i + seg, n)
        drift = np.linspace(0, rng.normal(0, SD.get(key, 0) * 0.15), j - i)
        vals[i:j] = level + drift
        i = j
    return vals


def main():
    rng = np.random.default_rng(SEED)
    t = pd.date_range(START, periods=N, freq=FREQ)

    # --- 1. Состав шихты: штабели (кусочно-постоянные) ---
    Cu_charge = piecewise_charge(rng, N, "Cu_charge")
    Fe_charge = piecewise_charge(rng, N, "Fe_charge")
    SiO2_charge = piecewise_charge(rng, N, "SiO2_charge")
    CaO_charge = piecewise_charge(rng, N, "CaO_charge")
    Zn_charge = piecewise_charge(rng, N, "Zn_charge")
    S_charge = piecewise_charge(rng, N, "S_charge")
    charge_moisture = B["charge_moisture"] + ar_process(rng, N, 0.4) \
        + 1.2 * np.sin(np.arange(N) * 2 * np.pi / 144)      # суточная сезонность

    # --- 2. Рычаги оператора (автокоррелированы; оператор частично реагирует на шихту) ---
    charge_flow = ar_process(rng, N, 3.0, base=B["charge_flow"])
    concentrate_flow = ar_process(rng, N, 4.0, base=B["concentrate_flow"]) \
        + 0.5 * (Cu_charge - B["Cu_charge"])                # богаче руда -> чуть больше
    coal_flow = ar_process(rng, N, 0.5, base=B["coal_flow"])
    coal_dust_flow = ar_process(rng, N, 0.3, base=B["coal_dust_flow"])
    # флюс: оператор добавляет под кремнезём/железо (шлаковый режим)
    addition_flow = ar_process(rng, N, 1.2, base=B["addition_flow"]) \
        + 0.6 * (Fe_charge - B["Fe_charge"]) - 0.4 * (SiO2_charge - B["SiO2_charge"])
    air_flow = ar_process(rng, N, 3.0, base=B["air_flow"])
    oxygen_flow = ar_process(rng, N, 6.0, base=B["oxygen_flow"])
    oxygen_concentration = np.clip(ar_process(rng, N, 4.5, base=B["oxygen_concentration"]),
                                   42, 63)
    natural_gas_flow = ar_process(rng, N, 60.0, base=B["natural_gas_flow"])

    # --- 3. Степень окисления (безразмерный ключевой драйвер) ---
    ox_raw = (0.55 * z(oxygen_flow, "oxygen_flow")
              + 0.45 * z(oxygen_concentration, "oxygen_concentration")
              + 0.20 * z(air_flow, "air_flow")
              - 0.35 * z(concentrate_flow, "concentrate_flow")
              - 0.20 * z(S_charge, "S_charge")
              - 0.15 * z(Fe_charge, "Fe_charge"))
    ox = ewm(ox_raw, 5)                                     # инерция ванны

    # --- 4. Продукты плавки ---
    # Штейн: матность растёт с окислением + с медью в руде (стационарно).
    Cu_matte = ewm(50.4 + 6.0 * ox + 0.45 * (Cu_charge - B["Cu_charge"]), 4) \
        + rng.normal(0, 0.6, N)
    Cu_matte = np.clip(Cu_matte, 44, 62)

    # Тепловой баланс расплава: топливо + экзотермия окисления − охлаждение дутьём.
    melt_temperature = ewm(
        1235 + 9.0 * ox
        + 0.045 * (natural_gas_flow - B["natural_gas_flow"])
        + 6.0 * (coal_flow - B["coal_flow"])
        + 1.2 * (S_charge - B["S_charge"])
        - 0.8 * (air_flow - B["air_flow"]), 6) + rng.normal(0, 1.5, N)

    # Шлаковый режим: баланс SiO2/Fe (управляется флюсом) — отклонение от оптимума.
    slag_ratio = (SiO2_charge + 0.20 * addition_flow) / Fe_charge
    ratio_dev = (slag_ratio - np.median(slag_ratio)) / np.std(slag_ratio)

    # Потери меди в шлак: U-образная по ox (оптимум!) + богатый штейн + плохой шлак + холод.
    Cu_slag = ewm(
        0.74
        + 0.42 * (ox - 0.0) ** 2                    # перекисление -> магнетит -> потери
        + 0.06 * ratio_dev ** 2                     # неоптимальный шлак -> вязкость
        + 0.05 * np.maximum(0, Cu_matte - 56)       # слишком богатый штейн
        + 0.05 * np.maximum(0, (1230 - melt_temperature) / 12), 4) \
        + rng.normal(0, 0.035, N)
    Cu_slag = np.clip(Cu_slag, 0.4, 1.7)

    # Шлак: железо, кремнезём, известь, цинк.
    Fe_slag = ewm(40.0 + 1.4 * (Fe_charge - B["Fe_charge"]) + 2.5 * ox
                  - 0.5 * (addition_flow - B["addition_flow"]), 4) + rng.normal(0, 0.4, N)
    SiO2_slag = ewm(29.6 + 1.6 * (SiO2_charge - B["SiO2_charge"])
                    + 0.7 * (addition_flow - B["addition_flow"]), 4) + rng.normal(0, 0.4, N)
    CaO_slag = ewm(2.95 + 1.8 * (CaO_charge - B["CaO_charge"])
                   + 0.05 * (addition_flow - B["addition_flow"]), 4) + rng.normal(0, 0.08, N)
    Zn_slag = ewm(4.95 + 1.1 * (Zn_charge - B["Zn_charge"]) - 0.4 * ox, 4) \
        + rng.normal(0, 0.15, N)

    # --- 5. Газовый тракт и котёл (несут след состава через SO2 и температуру) ---
    # SO2: окисленная сера. Сильно зависит от S в шихте и степени окисления -> состав виден.
    SO2_out = ewm(10.4 + 1.3 * (S_charge - B["S_charge"]) + 2.2 * ox
                  + 0.02 * (concentrate_flow - B["concentrate_flow"]), 5) \
        + rng.normal(0, 0.20, N)
    SO2_out = np.clip(SO2_out, 6.5, 14.5)

    offgas_flow = ewm(75000 + 300 * (air_flow - B["air_flow"])
                      + 60 * (oxygen_flow - B["oxygen_flow"])
                      + 8 * (natural_gas_flow - B["natural_gas_flow"]), 5) \
        + rng.normal(0, 250, N)
    offgas_temperature = ewm(959 + 0.55 * (melt_temperature - 1235)
                             + 0.03 * (natural_gas_flow - B["natural_gas_flow"]), 4) \
        + rng.normal(0, 3, N)
    boiler_inlet_gas_temperature = ewm(802 + 0.85 * (offgas_temperature - 959), 3) \
        + rng.normal(0, 3, N)
    caisson_temperature = 50.0 + 0.02 * (melt_temperature - 1235) \
        + ar_process(rng, N, 0.4) + rng.normal(0, 0.4, N)
    furnace_pressure = -57.0 - 0.002 * (offgas_flow - 75000) + rng.normal(0, 1.0, N)

    # Котёл-утилизатор: тепло газов -> пар.
    water_flow_circuits = 260 + 0.04 * (caisson_temperature - 50) * 100 \
        + ar_process(rng, N, 1.5) + rng.normal(0, 1.5, N)
    feedwater_flow_boiler = 131 + 0.05 * (boiler_inlet_gas_temperature - 802) \
        + ar_process(rng, N, 1.0) + rng.normal(0, 1.0, N)
    steam_pressure_drum = 29.1 + 0.01 * (boiler_inlet_gas_temperature - 802) \
        + rng.normal(0, 0.3, N)
    steam_temperature_drum = 306 + 0.08 * (boiler_inlet_gas_temperature - 802) \
        + rng.normal(0, 1.2, N)

    # Уровень ванны: баланс загрузки/выпуска (слабо управляем, медленный дрейф).
    melt_level = 1.516 + 0.0006 * (charge_flow - B["charge_flow"]) \
        + ar_process(rng, N, 0.008, phi=0.97) + rng.normal(0, 0.004, N)

    df = pd.DataFrame({
        "timestamp": t,
        "CaO_charge": CaO_charge, "CaO_slag": CaO_slag,
        "Cu_charge": Cu_charge, "Cu_matte": Cu_matte, "Cu_slag": Cu_slag,
        "Fe_charge": Fe_charge, "Fe_slag": Fe_slag,
        "SiO2_charge": SiO2_charge, "SiO2_slag": SiO2_slag,
        "Zn_charge": Zn_charge, "Zn_slag": Zn_slag,
        "charge_moisture": charge_moisture,
        "oxygen_concentration": oxygen_concentration, "oxygen_flow": oxygen_flow,
        "charge_flow": charge_flow, "concentrate_flow": concentrate_flow,
        "coal_flow": coal_flow, "coal_dust_flow": coal_dust_flow,
        "addition_flow": addition_flow, "air_flow": air_flow,
        "natural_gas_flow": natural_gas_flow, "melt_temperature": melt_temperature,
        "offgas_temperature": offgas_temperature,
        "boiler_inlet_gas_temperature": boiler_inlet_gas_temperature,
        "caisson_temperature": caisson_temperature, "furnace_pressure": furnace_pressure,
        "offgas_flow": offgas_flow, "SO2_out": SO2_out, "melt_level": melt_level,
        "water_flow_circuits": water_flow_circuits,
        "feedwater_flow_boiler": feedwater_flow_boiler,
        "steam_pressure_drum": steam_pressure_drum,
        "steam_temperature_drum": steam_temperature_drum, "S_charge": S_charge,
    })

    df.to_csv("data/dataset.csv", index=False)
    print(f"Сгенерировано {len(df)} строк -> data/dataset.csv")
    print("\nКлючевые средние (цель -> факт):")
    for k, tgt in [("Cu_matte", 50.4), ("Cu_slag", 0.92), ("melt_temperature", 1235),
                   ("SO2_out", 10.4), ("Fe_slag", 40), ("SiO2_slag", 29.6),
                   ("melt_level", 1.516), ("oxygen_flow", 201)]:
        print(f"  {k:18} {tgt:>8} -> {df[k].mean():8.3f}  (std {df[k].std():.3f})")


if __name__ == "__main__":
    main()
