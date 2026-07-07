"""Единая конфигурация проекта: роли колонок, границы рычагов, веса целевой функции.

Разделение колонок отражает РЕАЛЬНУЮ боль с производства (со слов организаторов):
онлайн-сигналы доступны сразу с датчиков АСУ ТП, а химсостав шихты и продуктов
(штейн/шлак) приходит из лаборатории с задержкой в смену. Цель системы — убрать
этот информационный лаг.
"""
from __future__ import annotations

TIMESTAMP = "timestamp"

# --- Онлайн-сигналы: доступны в реальном времени с датчиков/расходомеров ---
# Управляемые рычаги оператора (их крутит оптимизатор):
LEVERS = [
    "charge_flow",
    "concentrate_flow",
    "coal_flow",
    "coal_dust_flow",
    "addition_flow",
    "air_flow",
    "oxygen_flow",
    "oxygen_concentration",
    "natural_gas_flow",
]

# Онлайн-состояние печи и котла (меряется, но не задаётся напрямую):
ONLINE_STATE = [
    "melt_temperature",
    "melt_level",
    "caisson_temperature",
    "furnace_pressure",
    "offgas_temperature",
    "offgas_flow",
    "SO2_out",
    "boiler_inlet_gas_temperature",
    "water_flow_circuits",
    "feedwater_flow_boiler",
    "steam_pressure_drum",
    "steam_temperature_drum",
    "charge_moisture",  # допущение: онлайн-влагомер; уточнить у организаторов
]

# Все онлайн-признаки (вход soft-sensor'ов):
ONLINE = LEVERS + ONLINE_STATE

# --- Лабораторные показатели: приходят с лагом (смена). Это таргеты. ---
# Состав шихты — ГЛАВНАЯ боль (ночную шихту узнают только к утру):
CHARGE_COMPOSITION = [
    "Cu_charge",
    "Zn_charge",
    "CaO_charge",
    "SiO2_charge",
    "Fe_charge",
    "S_charge",
]

# Продукты плавки (рентген/химанализ):
PRODUCTS = [
    "Cu_matte",   # содержание Cu в штейне
    "Cu_slag",    # содержание Cu в шлаке — ПОТЕРИ меди, минимизируем
    "Zn_slag",
    "CaO_slag",
    "SiO2_slag",
    "Fe_slag",
]

LAB = CHARGE_COMPOSITION + PRODUCTS

# --- Целевая функция оптимизатора (слой 3) ---
# Минимизируем потери меди в шлак + удельный расход ресурсов.
# Веса — стартовые, подбираются под здравый смысл на демо.
# Веса — приоритеты ПОСЛЕ нормировки на масштаб (см. optimize.objective_value).
# Cu_slag доминирует; ресурсы урезаются вторично, когда это почти бесплатно по меди.
OBJECTIVE_MINIMIZE = {
    "Cu_slag": 1.0,          # главный приоритет — потери меди
    "oxygen_flow": 0.1,
    "natural_gas_flow": 0.1,
    "coal_flow": 0.1,
    "coal_dust_flow": 0.1,
}

# Технологические ограничения (наполним фактическими диапазонами после EDA).
# melt_temperature держим в рабочем окне, Cu_matte — кондиция штейна, SO2 — экология.
CONSTRAINTS = {
    "melt_temperature": {"min": 1210.0, "max": 1260.0},
    "Cu_matte": {"min": 45.0},
    "SO2_out": {"max": 13.0},
    "melt_level": {"min": 1.48, "max": 1.55},
}

DATA_PATH = "data/dataset.csv"
