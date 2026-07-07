# Прогноз и оптимизация режима печи Ванюкова

Система убирает информационный лаг оператора: по онлайн-датчикам оценивает состав шихты
и штейн (не дожидаясь лаборатории), ловит смену штабеля и отклонения режима, советует
режим для снижения потерь меди в шлак и расхода ресурсов.

## Структура

```
data/dataset.csv        # основной датасет (сгенерирован src/generate.py)
src/                    # ядро (модели и логика)
  config.py             #   роли колонок, границы, целевая функция
  data.py               #   загрузка + хронологический сплит
  generate.py           #   физический генератор датасета
  soft_sensor.py        #   soft-sensor (штейн, состав)
  twin.py               #   цифровой двойник f(рычаги,состав)->продукты
  optimize.py           #   Optuna-оптимизатор режима
  anomaly.py            #   батч-анализ детекторов (валидация)
  detectors.py          #   онлайн-детекторы (stateful, для API)
  pipeline.py           #   сквозной поток end-to-end
  train.py              #   обучить всё -> models/*.pkl
api/                    # FastAPI-сервис
  main.py               #   эндпоинты
  schemas.py            #   pydantic вход/выход
  service.py            #   загрузка моделей + обёртки
models/*.pkl            # обученные артефакты
docs/AGENT_API_GUIDE.md # гайд для ИИ-агента (интерфейс + симуляция)
reports/                # метрики, графики, findings
deprecated/             # исследовательские скрипты (EDA, диагностика)
```

## Установка

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
brew install libomp        # LightGBM на macOS
```

## Запуск

```bash
# 1. (пере)сгенерировать датасет
PYTHONPATH=src .venv/bin/python src/generate.py

# 2. обучить модели -> models/*.pkl
PYTHONPATH=src .venv/bin/python src/train.py

# 3. проверить сквозной поток в терминале
PYTHONPATH=src .venv/bin/python src/pipeline.py

# 4. поднять API
PYTHONPATH=src .venv/bin/uvicorn api.main:app --port 8000
#    Swagger: http://localhost:8000/docs
```

## Результаты (на сгенерированном датасете)

- Soft-sensor штейна: между лаб-замерами точнее на **83%**
- Оптимизатор: потери меди `Cu_slag` **−50%** на проблемных режимах, ресурсы −6%, 100% в допуске
- Детектор смены штабеля: recall **100%**, опережение лаборатории **7.3 ч**
- Детектор отклонений режима: recall 83%

Метрики демонстрируют, что метод восстанавливает заложенную физику; для продакшена нужна
калибровка на реальных данных. Подробности — `reports/findings.md`.

## Для интеграции

Гайд для агента, строящего UI и симуляцию датчиков: **`docs/AGENT_API_GUIDE.md`**.
