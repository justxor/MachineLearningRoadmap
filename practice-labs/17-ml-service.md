# 🚀 Лаба 17: ML-сервис на FastAPI + Docker + CI 🟡

## Цель

Упаковать ML-модель в сервис: REST API, контейнеризация, автопроверки и деплой. Понять разницу между моделью в .ipynb и сервисом в проде.

## Компоненты

- FastAPI + Pydantic схемы.
- Dockerfile (мультистейджный).
- GitHub Actions: lint + test + build + push image.
- /healthz, /readyz endpoints.
- Prometheus metrics: счётчики запросов, latency.

## Минимальный пайплайн

1. Обёрнуть модель (любую из лаб 01–11) в endpoint /predict.
2. Input validation через Pydantic.
3. Logging request_id, latency, errors.
4. Dockerfile + docker-compose.
5. Unit + integration tests (pytest).
6. CI: полный пайплайн при push/PR.
7. Сборка и push image в ghcr.io.

## Код: FastAPI endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, time

app = FastAPI()
model = joblib.load("model.pkl")

class Req(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(r: Req):
    t0 = time.time()
    y = model.predict([r.features])[0]
    return {"prediction": float(y), "latency_ms": (time.time()-t0)*1000}
```

## Dockerfile

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Метрики и проверки

- p50/p95 latency (в логах).
- RPS при nagrузке (locust/k6).
- Coverage tests >70%.
- Image size <500MB.
- Cold start <5с.

## Расширения

- Batching: собирать запросы в batch для GPU.
- gRPC вариант.
- Async + queue (Celery/Redis).
- BentoML / Ray Serve как альтернатива.
- Helm chart для Kubernetes deploy.

## Критерии приёмки

- [ ] /predict работает локально и в docker.
- [ ] CI зелёный на основной ветке.
- [ ] Image опубликован в ghcr.
- [ ] Tests покрывают endpoint + edge cases.
- [ ] README с примером curl.
- [ ] Нагрузочный тест: ≥5 RPS и p95 <200мс.

## Анти-паттерны

- ❌ Модель загружается на каждый запрос.
- ❌ Нет input validation — сервис падает на мусорных данных.
- ❌ Logging в stdout плохо структурировано (разберите json-logger).
- ❌ Секреты в git.
- ❌ Без healthcheck — k8s не знает живой ли pod.

---

[← Назад к Practice Labs](./README.md)
