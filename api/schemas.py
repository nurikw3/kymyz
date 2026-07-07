"""Pydantic-схемы входа/выхода API печи Ванюкова.

OnlineSnapshot — то, что реально даёт АСУ ТП в реальном времени (22 датчика/расхода).
Это единственный обязательный вход для большинства эндпоинтов.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class OnlineSnapshot(BaseModel):
    """Онлайн-срез датчиков (доступно в реальном времени). Все значения — float."""
    # --- рычаги оператора (дутьё и расходы) ---
    charge_flow: float = Field(..., description="расход шихты, т/ч")
    concentrate_flow: float = Field(..., description="расход концентрата, т/ч")
    coal_flow: float = Field(..., description="расход угля, т/ч")
    coal_dust_flow: float = Field(..., description="расход угольной пыли, т/ч")
    addition_flow: float = Field(..., description="расход флюса/подшихтовки, т/ч")
    air_flow: float = Field(..., description="расход воздуха, тыс. м³/ч")
    oxygen_flow: float = Field(..., description="удельный расход кислорода")
    oxygen_concentration: float = Field(..., description="концентрация O2 в дутье, %")
    natural_gas_flow: float = Field(..., description="расход природного газа, м³/ч")
    # --- состояние печи и котла ---
    melt_temperature: float = Field(..., description="температура расплава, °C")
    melt_level: float = Field(..., description="уровень расплава, м")
    caisson_temperature: float = Field(..., description="температура кессонов, °C")
    furnace_pressure: float = Field(..., description="разрежение в печи, Па")
    offgas_temperature: float = Field(..., description="температура отходящих газов, °C")
    offgas_flow: float = Field(..., description="расход отходящих газов, м³/ч")
    SO2_out: float = Field(..., description="концентрация SO2 на выходе, %")
    boiler_inlet_gas_temperature: float = Field(..., description="температура газов на входе котла, °C")
    water_flow_circuits: float = Field(..., description="расход воды в контурах охлаждения")
    feedwater_flow_boiler: float = Field(..., description="расход питательной воды на барабан")
    steam_pressure_drum: float = Field(..., description="давление пара в барабане")
    steam_temperature_drum: float = Field(..., description="температура пара в барабане, °C")
    charge_moisture: float = Field(..., description="влажность шихты, %")

    model_config = {"extra": "forbid"}


class ChargeComposition(BaseModel):
    """Состав шихты (лабораторный). Используется как last_lab_charge (прошлая смена)."""
    Cu_charge: float = Field(..., description="Cu в шихте, %")
    Zn_charge: float = Field(..., description="Zn в шихте, %")
    CaO_charge: float = Field(..., description="CaO в шихте, %")
    SiO2_charge: float = Field(..., description="SiO2 в шихте, %")
    Fe_charge: float = Field(..., description="Fe в шихте, %")
    S_charge: float = Field(..., description="сера в шихте, %")

    model_config = {"extra": "forbid"}


class LeverSettings(BaseModel):
    """9 управляемых рычагов оператора (для прямого прогноза двойником)."""
    charge_flow: float
    concentrate_flow: float
    coal_flow: float
    coal_dust_flow: float
    addition_flow: float
    air_flow: float
    oxygen_flow: float
    oxygen_concentration: float
    natural_gas_flow: float

    model_config = {"extra": "forbid"}


# ---------- входные обёртки ----------

class ChargeRequest(BaseModel):
    online: OnlineSnapshot
    last_lab_charge: ChargeComposition | None = Field(
        None, description="последний лаб-замер состава (прошлая смена); опционально")


class TwinRequest(BaseModel):
    levers: LeverSettings
    charge: ChargeComposition


class OptimizeRequest(BaseModel):
    online: OnlineSnapshot
    last_lab_charge: ChargeComposition | None = None
    n_trials: int = Field(200, ge=50, le=1000, description="итераций Optuna")


class StreamRequest(BaseModel):
    online: OnlineSnapshot
    last_lab_charge: ChargeComposition | None = None
    optimize: bool = Field(True, description="считать ли рекомендацию режима (медленнее)")
