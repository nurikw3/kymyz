# API печи Ванюкова — FastAPI + LightGBM
FROM python:3.11-slim

WORKDIR /app

# libgomp1 — OpenMP-рантайм для LightGBM на Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код, данные и готовые модели (models/*.pkl из репозитория)
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
