# 20. Мониторинг и MLOps для CV

CV-модели в проде живут не вечно. Камеры меняются, освещение, мода, объекты. Drift detection и retraining — обязательны.

## Что мониторить

### Service health
- Latency p50/p95/p99.
- Throughput (RPS).
- Error rate (5xx, timeouts).
- GPU/CPU/memory utilization.
- Queue depth.

### Model quality
- Confidence distribution (по дням, по сегментам).
- % low-confidence predictions.
- Class distribution (что предсказывает).
- Disagreement с user feedback / ground truth.

### Data drift
- Input image statistics: brightness, contrast, sharpness, dominant colors.
- Embedding drift (через CLIP/DINOv2): новые типы картинок.
- Resolution distribution, aspect ratio.
- Failure rate в preprocessing (corrupted images).

## Drift detection для изображений

### Statistical drift

Сравниваем статистики train vs production:
- Mean/std каждого канала.
- Гистограммы.
- PSI (Population Stability Index) на каждой статистике.

### Embedding drift

Сравниваем CLIP/DINOv2 embeddings train batch vs production batch:
- MMD (Maximum Mean Discrepancy).
- Kolmogorov-Smirnov test по каждой dimension.
- Wasserstein distance.

```python
import torch
from transformers import CLIPModel

clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

train_embeds = clip.get_image_features(train_images)  # (N_train, 512)
prod_embeds = clip.get_image_features(prod_images)    # (N_prod, 512)

# MMD
def mmd(x, y, sigma=1.0):
    xx = torch.exp(-torch.cdist(x, x)**2 / (2*sigma**2)).mean()
    yy = torch.exp(-torch.cdist(y, y)**2 / (2*sigma**2)).mean()
    xy = torch.exp(-torch.cdist(x, y)**2 / (2*sigma**2)).mean()
    return xx + yy - 2*xy

drift_score = mmd(train_embeds, prod_embeds)
```

### Per-class drift

Watch for: «класс X стал появляться чаще на 30%». Может быть сезонность, может быть реальный drift.

## Tools

### Evidently AI

Open-source data quality and drift monitoring. Поддерживает табличные данные и embeddings.

```python
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric

report = Report(metrics=[EmbeddingsDriftMetric()])
report.run(reference_data=train_df, current_data=prod_df)
```

### Arize / Fiddler / WhyLabs

Commercial monitoring platforms. Имеют image support, dashboard, alerting.

### Weights & Biases

Хорош для experiment tracking + lightweight prod monitoring.

### Prometheus + Grafana

DIY. Свои метрики + дашборды. Стандарт для technical metrics.

## Logging predictions

Лог каждого prediction (или sample) для оффлайн анализа:

```python
{
    "request_id": "uuid",
    "timestamp": "2026-01-15T10:30:00Z",
    "image_hash": "sha256...",
    "model_version": "v2.3.1",
    "predictions": [{"class": "cat", "confidence": 0.87, "bbox": [...]}],
    "input_stats": {"width": 1920, "height": 1080, "brightness": 142},
    "latency_ms": 45,
}
```

**Sampling:** не каждый запрос (дорого), 1-10% случайно.

## Human-in-the-loop

### Review queues

Low-confidence predictions → human review queue. Human verdicts → new training data.

```python
if max_confidence < 0.7:
    send_to_review_queue(image, predictions)
```

### Feedback loops

UI с «нет, это не X»: пользовательский feedback → подтверждение/correction → новые labels.

**Подвох:** users часто кликают «неправильно» по другим причинам. Не доверять 1-1, нужна агрегация.

## Retraining strategy

### Triggers

- **Calendar:** раз в месяц/квартал.
- **Drift:** PSI > threshold.
- **Quality:** measured accuracy < SLA.
- **Volume:** накопилось N новых labeled samples.

### Continuous training pipeline

```
[new data] → [auto-label/review] → [aggregate train set] → 
[retrain] → [eval vs prev] → [shadow deploy] → 
[A/B test] → [promote или rollback]
```

### Catastrophic forgetting

При retrain на новых данных модель может «забыть» старые сценарии.

**Лекарства:**
- Mix новых и старых данных в train.
- Replay buffer: всегда include самые ценные старые examples.
- Knowledge distillation от старой модели.

## Shadow deployment

Новая модель получает реальный трафик, но её ответы **не используются** (только logged). Сравниваем с продовой:
- Agreement rate.
- Where they disagree?
- Latency, errors.

Минимум 1-2 недели shadow перед canary.

## A/B testing для CV

Аналогично рекомендациям/поиску:
- Random assignment пользователей на A или B.
- Long-term metrics: user satisfaction, conversion, retention.
- Beware: CV impact часто не в immediate clicks, а в long-term experience.

## CI/CD для CV

### Training в CI

При каждом merge — auto-train на small dataset → sanity-check accuracy не упала ниже baseline.

### Model versioning

- **MLflow Model Registry** — стандарт.
- **W&B Artifacts** — альтернатива.
- **DVC** — для версионирования и данных, и моделей.

### Eval в CI

Pull request с изменением модели → automated eval на golden dataset → comment в PR с метриками.

## Data versioning

CV датасеты огромные. Подходы:
- **DVC + S3:** стандарт.
- **lakeFS** — git-like для data lake.
- **Hugging Face Datasets:** для open data.

## Антипаттерны

- **Только service-level monitoring без data drift.** Сервис работает, метрика валится.
- **Hold-out eval set, который не обновляется.** Через год не отражает реальность.
- **Push-only changes.** Без auto-revert при деградации — рискованно.
- **Не логировать predictions.** Когда что-то ломается — не понять, что именно.
- **Training-prod skew** (preprocessing разный). Стандартизируйте через pipeline as a service.

## Задания

1. Настроить Evidently AI для drift detection на embedding входных картинок.
2. Реализовать low-confidence routing в review queue. Замерить % triggered и validation by humans.
3. Сделать shadow deployment скрипт: новая модель параллельно с прод, сравнить agreement.
4. Построить retrain pipeline через Airflow: триггер по PSI, eval, manual approval, deploy.
5. Реализовать continuous monitoring dashboard: latency, throughput, confidence distribution, drift score.
6. Сравнить эффект catastrophic forgetting: retrain только на новых vs mix старых+новых.

## Чек-лист

- [ ] Понимаю service-level vs model-level vs data-level мониторинг.
- [ ] Умею детектить data drift через embeddings.
- [ ] Знаю Evidently, Arize, Prometheus.
- [ ] Понимаю shadow deployment, A/B для CV.
- [ ] Знаю про catastrophic forgetting и его mitigation.

## Дальше

➡️ [21-safety-privacy.md](./21-safety-privacy.md) — безопасность и privacy в CV.
