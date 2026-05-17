# 18. От ноутбука к сервису: API, Docker

Модель, которая живёт в ipynb, бизнесу не полезна. Надо уметь обёртывать предикт в API и поднимать его в Docker.

## Минимум

- FastAPI или LitServe.
- Docker + multi-stage build.
- pydantic для схем входа/выхода.
- pickle / joblib / onnx для весов.
- gunicorn / uvicorn + workers.

## Минимальный FastAPI сервис

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("model.joblib")
app = FastAPI()

class Req(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(r: Req):
    return {"score": float(model.predict_proba([r.features])[0, 1])}
```

## Dockerfile

```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Что важно в проде

- Схема фичей валидируется.
- Логирование входов/выходов (без PII).
- Health/readiness endpoints.
- Версия модели в ответе.
- Graceful shutdown.
- Таймауты и лимиты.

## Антипаттерны

- Latex обязательных фичей не валидируется.
- Модель загружается на каждый запрос.
- Нет версионирования модели.
- Обработка фичей в коде сервиса, который не совпадает с трейн-пайплайном.

## Практика

1. Заверните модель в FastAPI.
2. Напишите Dockerfile, поднимите локально.
3. Добавьте /health и /version endpoints.
4. Напишите 5 pytest-тестов на API.
5. Оцените latency через locust / wrk.
6. Соберите docker-compose с Postgres и Redis.

## Чек-лист

- [ ] API поднимается в docker.
- [ ] Есть валидация входа.
- [ ] Есть версия модели.

[← 17](17-interpretation.md) • [→ 19. MLOps](19-mlops.md)
