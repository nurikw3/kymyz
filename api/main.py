"""FastAPI-сервис системы прогноза и оптимизации режима печи Ванюкова.

Запуск (из корня проекта):
    PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --port 8000

Интерактивная документация: http://localhost:8000/docs
Полный гайд для интеграции/симуляции: docs/AGENT_API_GUIDE.md
"""
from __future__ import annotations

from fastapi import FastAPI

from api.schemas import (OnlineSnapshot, ChargeRequest, TwinRequest,
                         OptimizeRequest, StreamRequest)
from api.service import FurnaceService

app = FastAPI(
    title="Vanyukov Furnace API",
    version="1.0",
    description="Оценка шихты/штейна, цифровой двойник, оптимизация режима, детекторы. "
                "Все входы — онлайн-датчики (АСУ ТП). Гайд: docs/AGENT_API_GUIDE.md",
)
svc = FurnaceService()


@app.get("/health", tags=["meta"])
def health():
    """Проверка готовности сервиса и загрузки моделей."""
    return {"status": "ok", "models": ["pipeline", "anomaly_detector", "shift_detector"]}


@app.get("/schema", tags=["meta"])
def schema():
    """Списки полей: онлайн-входы, рычаги, состав шихты, наблюдаемость онлайн."""
    return svc.schema()


@app.get("/ranges", tags=["meta"])
def ranges():
    """min/median/max по онлайн-входам — для настройки симулятора датчиков."""
    return svc.ranges()


@app.get("/sample", tags=["meta"])
def sample(n: int = 1, tail: bool = True):
    """Реальные онлайн-срезы из датасета для проигрывания в симуляции (n срезов)."""
    return {"samples": svc.sample(n=n, tail=tail)}


@app.post("/soft-sensor/matte", tags=["soft-sensor"])
def soft_sensor_matte(online: OnlineSnapshot):
    """Оценка Cu в штейне онлайн (без лаборатории). Вход: срез датчиков → выход: Cu_matte."""
    return svc.estimate_matte(online.model_dump())


@app.post("/soft-sensor/charge", tags=["soft-sensor"])
def soft_sensor_charge(req: ChargeRequest):
    """Оценка состава шихты. Сера — онлайн; остальное — по last_lab_charge (персистентность)."""
    return svc.estimate_charge(req.online.model_dump(),
                               req.last_lab_charge.model_dump() if req.last_lab_charge else None)


@app.post("/twin/predict", tags=["twin"])
def twin_predict(req: TwinRequest):
    """Цифровой двойник: рычаги + состав → прогноз продуктов/состояния (Cu_slag, Cu_matte…)."""
    return svc.twin_predict(req.levers.model_dump(), req.charge.model_dump())


@app.post("/optimize", tags=["optimize"])
def optimize_regime(req: OptimizeRequest):
    """Рекомендация режима: минимизировать Cu_slag + ресурсы при техограничениях."""
    return svc.optimize_regime(req.online.model_dump(),
                               req.last_lab_charge.model_dump() if req.last_lab_charge else None,
                               n_trials=req.n_trials)


@app.post("/detect/anomaly", tags=["detect"])
def detect_anomaly(online: OnlineSnapshot):
    """Отклонение режима: факт vs эталон. Выход: anomaly, channel, z."""
    return svc.detect_anomaly(online.model_dump())


@app.post("/stream/step", tags=["stream"])
def stream_step(req: StreamRequest):
    """Полный проход для одного среза потока (для симуляции датчиков): оценки + детекторы
    + рекомендация. Детектор смены штабеля stateful — вызывать последовательно."""
    return svc.stream_step(req.online.model_dump(),
                           req.last_lab_charge.model_dump() if req.last_lab_charge else None,
                           do_optimize=req.optimize)


@app.post("/stream/reset", tags=["stream"])
def stream_reset():
    """Сбросить буфер потока (перед новой симуляцией)."""
    return svc.reset_stream()
