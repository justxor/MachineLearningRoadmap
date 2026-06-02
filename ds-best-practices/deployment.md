# 🚀 Deployment — Деплой и Продакшн

Паттерны и сниппеты для перевода ML моделей в продакшн.

---

## 1. FastAPI ML Сервис

```python
# pip install fastapi uvicorn pydantic joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="ML Model API",
    description="REST API для ML модели",
    version="1.0.0",
)

# Загрузка модели при старте
@app.on_event("startup")
async def load_model():
    global model, scaler
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Model loaded successfully")

# Схема входных данных
class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Возраст")
    salary: float = Field(..., ge=0, description="Зарплата")
    experience: float = Field(..., ge=0, le=50, description="Опыт в годах")
    city: str = Field(..., description="Город")

    @validator('city')
    def city_must_be_valid(cls, v):
        allowed_cities = ['moscow', 'spb', 'other']
        if v.lower() not in allowed_cities:
            raise ValueError(f"city must be one of {allowed_cities}")
        return v.lower()

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    prediction_time_ms: float
    model_version: str = "1.0.0"

# Эндпоинты
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        # Подготовка данных
        features = pd.DataFrame([request.dict()])
        features_scaled = scaler.transform(features[['age', 'salary', 'experience']])

        # Предсказание
        prediction = model.predict(features_scaled)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0, 1]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Prediction: {prediction:.4f} | {elapsed_ms:.1f}ms")

        return PredictionResponse(
            prediction=float(prediction),
            probability=float(proba) if proba else None,
            prediction_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "version": "1.0.0",
        "features": ['age', 'salary', 'experience', 'city'],
    }

# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Dockerfile для ML сервиса

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Создание пользователя без root прав
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/best_model.pkl
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## 3. Мониторинг дрейфа данных

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict

def detect_data_drift(reference_df: pd.DataFrame,
                       current_df: pd.DataFrame,
                       threshold: float = 0.05) -> Dict:
    """
    Обнаружение дрейфа данных между референсным и текущим датасетами.
    Использует KS-тест для числовых и Chi-Square для категориальных.
    """
    drift_report = {}

    for col in reference_df.columns:
        if reference_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Числовые — KS тест
            stat, p_value = ks_2samp(
                reference_df[col].dropna(),
                current_df[col].dropna()
            )
            drift_report[col] = {
                'type': 'numeric',
                'test': 'KS',
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': p_value < threshold,
                'ref_mean': reference_df[col].mean(),
                'cur_mean': current_df[col].mean(),
                'mean_change_pct': abs(current_df[col].mean() - reference_df[col].mean()) / (abs(reference_df[col].mean()) + 1e-8) * 100
            }
        else:
            # Категориальные — Chi-Square тест
            ref_counts = reference_df[col].value_counts()
            cur_counts = current_df[col].value_counts()
            all_cats = set(ref_counts.index) | set(cur_counts.index)
            contingency = pd.DataFrame({
                'ref': [ref_counts.get(c, 0) for c in all_cats],
                'cur': [cur_counts.get(c, 0) for c in all_cats],
            })
            chi2, p_value, _, _ = chi2_contingency(contingency)
            drift_report[col] = {
                'type': 'categorical',
                'test': 'Chi-Square',
                'statistic': chi2,
                'p_value': p_value,
                'drift_detected': p_value < threshold,
                'new_categories': set(cur_counts.index) - set(ref_counts.index),
                'missing_categories': set(ref_counts.index) - set(cur_counts.index),
            }

    # Сводка
    drifted = [col for col, info in drift_report.items() if info['drift_detected']]
    print(f"Data Drift Report:")
    print(f"  Total features: {len(drift_report)}")
    print(f"  Features with drift: {len(drifted)} ({len(drifted)/len(drift_report)*100:.1f}%)")
    if drifted:
        print(f"  Drifted features: {drifted}")

    return drift_report
```

---

## 4. Батчевые предсказания

```python
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

def batch_predict(input_path: str, output_path: str, model_path: str,
                   batch_size: int = 10000, chunksize: int = 50000):
    """Батчевые предсказания на больших датасетах"""
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    # Читаем по чанкам
    for chunk in tqdm(pd.read_csv(input_path, chunksize=chunksize)):
        # Предсказания
        predictions = model.predict(chunk)

        result = pd.DataFrame({
            'id': chunk.index,
            'prediction': predictions,
        })

        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(chunk)
            result['probability'] = probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)

        all_predictions.append(result)

    # Сохраняем результат
    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved: {len(final_df)} rows → {output_path}")

    return final_df
```

---

## 5. A/B тест для ML моделей

```python
from scipy import stats

def ab_test_models(predictions_a: np.ndarray, predictions_b: np.ndarray,
                    ground_truth: np.ndarray, metric_func,
                    n_bootstrap: int = 10000, alpha: float = 0.05) -> dict:
    """
    Статистический A/B тест для сравнения двух моделей.
    Используем Bootstrap для оценки значимости разницы.
    """
    score_a = metric_func(ground_truth, predictions_a)
    score_b = metric_func(ground_truth, predictions_b)

    # Bootstrap
    rng = np.random.RandomState(42)
    n = len(ground_truth)
    delta_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        s_a = metric_func(ground_truth[idx], predictions_a[idx])
        s_b = metric_func(ground_truth[idx], predictions_b[idx])
        delta_scores.append(s_b - s_a)

    delta_scores = np.array(delta_scores)
    ci_lower = np.percentile(delta_scores, alpha / 2 * 100)
    ci_upper = np.percentile(delta_scores, (1 - alpha / 2) * 100)

    # p-value через Bootstrap
    p_value = np.mean(delta_scores <= 0) * 2  # двусторонний

    result = {
        'score_a': score_a,
        'score_b': score_b,
        'delta': score_b - score_a,
        'delta_pct': (score_b - score_a) / abs(score_a) * 100,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'winner': 'B' if (score_b > score_a and p_value < alpha) else ('A' if (score_a > score_b and p_value < alpha) else 'No winner'),
    }

    print(f"A/B Test Results (α={alpha}):")
    print(f"  Model A: {score_a:.4f}")
    print(f"  Model B: {score_b:.4f}")
    print(f"  Delta: {result['delta']:+.4f} ({result['delta_pct']:+.1f}%)")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Winner: {result['winner']}")

    return result
```

---

## 6. Чеклист перед деплоем

```markdown
## Pre-Deployment Checklist

### Данные
- [ ] Распределение трейна ≈ продакшн данным
- [ ] Нет утечки данных (data leakage)
- [ ] Описаны все признаки и их источники
- [ ] Задокументированы пропуски и аномалии

### Модель
- [ ] CV score стабилен (малое std)
- [ ] Test score близок к CV score (нет переобучения)
- [ ] Вероятности откалиброваны (если нужны)
- [ ] Проверена работа на граничных случаях

### Код
- [ ] Unit tests для preprocessing
- [ ] Integration tests для API
- [ ] Зависимости зафиксированы (requirements.txt / poetry.lock)
- [ ] Модель и scaler сохранены в одном артефакте

### Мониторинг
- [ ] Настроено логирование предсказаний
- [ ] Алерты на дрейф данных
- [ ] Мониторинг метрик (latency, errors, predictions distribution)
- [ ] Установлен baseline для сравнения

### Документация
- [ ] README с инструкцией запуска
- [ ] API документация (автоматически через FastAPI/Swagger)
- [ ] Описание входных/выходных данных
- [ ] Контактная информация владельца модели
```
