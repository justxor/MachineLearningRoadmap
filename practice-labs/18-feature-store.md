# 🏪 Лаба 18: Feature store + retraining pipeline 🔴

## Цель

Построить feature store и автоматический пайплайн переобучения модели по расписанию или по событиям. Понять разницу между online/offline фичами и как избежать training/serving skew.

## Стек

- Feast или Hopsworks как feature store.
- Airflow / Prefect / Dagster для orchestration.
- MLflow / W&B для experiment tracking.
- DVC для версии данных.
- S3/MinIO для offline store.
- Redis для online store.

## Минимальный пайплайн

1. Определить entities и feature views в Feast.
2. ETL: сырые данные → features (Spark/SQL/Python).
3. Материализация в online store.
4. Training pipeline: pull historical features (point-in-time correct) → train → log в MLflow.
5. Auto-trigger: по cron или по дрифту.
6. Deployment: новая модель → staging → canary → prod.

## Feast: пример feature view

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

user = Entity(name="user_id", join_keys=["user_id"])
src = FileSource(path="data/user_features.parquet", timestamp_field="event_ts")

user_features = FeatureView(
  name="user_stats",
  entities=[user],
  ttl=timedelta(days=7),
  schema=[Field(name="avg_purchase", dtype=Float32), Field(name="orders_30d", dtype=Int64)],
  source=src,
)
```

## Критические принципы

- Point-in-time correctness — никаких future leaks.
- Online/offline parity — один код для train и inference.
- Idempotent ETL — можно перезапускать.
- Backfill: посчитать фичи для истории.

## Метрики

- Latency online feature retrieval (p99 <50мс).
- ETL freshness (lag в часах).
- Training/serving skew — расхождения фичей на train vs prod.
- Pipeline success rate.

## Расширения

- Streaming features (Kafka + Flink).
- Feature monitoring (распределения, null rate).
- Auto-rollback при падении метрик.
- Shadow deployment.
- Feature catalog с овнерами.

## Критерии приёмки

- [ ] Feast (или аналог) поднят, materialize работает.
- [ ] Training pipeline в Airflow/Prefect.
- [ ] MLflow experiments ведутся.
- [ ] Online retrieval p99 <50мс.
- [ ] Point-in-time correctness доказана (унит-тест).
- [ ] Runbook для перезапуска backfill.

## Анти-паттерны

- ❌ Разный код для train и inference фичей.
- ❌ Нет версии данных.
- ❌ ETL не idempotent — двойные запуски ломают данные.
- ❌ Переобучение «каждую ночь» без проверки качества.
- ❌ Деплой без staging/canary.

---

[← Назад к Practice Labs](./README.md)
