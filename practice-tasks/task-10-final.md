# 🏆 Задача 10 — Финальный Проект: End-to-End ML

**Сложность:** ⭐⭐⭐⭐⭐ | **Время:** 6–10 часов
**Разбор:** [solution-10-final.md](../practice-solutions/solution-10-final.md)

---

## 📋 Описание

Самостоятельный end-to-end проект, объединяющий все навыки курса. Ты создаёшь полноценный ML продукт — от сырых данных до деплоя REST API.

---

## 🎯 Бизнес-задача

**Телеком компания** хочет предсказывать отток клиентов (churn) заблаговременно, чтобы предложить им персональные удержательные акции. Бюджет на акции ограничен — нельзя предлагать их всем подряд.

**Бизнес-ограничение:** обрабатывать не более 10% клиентской базы в месяц, при этом поймать максимум реальных "уходящих".

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 7000

df = pd.DataFrame({
    'customer_id':        range(n),
    'tenure_months':      np.random.randint(1, 72, n),
    'monthly_charges':    np.abs(np.random.normal(65, 30, n)),
    'total_charges':      np.abs(np.random.normal(2000, 1800, n)),
    'contract_type':      np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.2]),
    'payment_method':     np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'internet_service':   np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.44, 0.21]),
    'phone_service':      np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
    'online_security':    np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.5, 0.21]),
    'tech_support':       np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.5, 0.21]),
    'streaming_tv':       np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.41, 0.21]),
    'num_complaints':     np.random.randint(0, 5, n),
    'avg_monthly_usage':  np.abs(np.random.normal(50, 20, n)),
    'support_calls':      np.random.randint(0, 8, n),
    'gender':             np.random.choice(['Male', 'Female'], n),
    'senior_citizen':     np.random.binomial(1, 0.16, n),
    'partner':            np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
    'dependents':         np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
    'churn':              np.random.binomial(1, 0.265, n),
})

# Добавляем реалистичность и проблемы
df.loc[np.random.choice(n, 150), 'total_charges'] = np.nan
df.loc[np.random.choice(n, 80), 'monthly_charges'] = np.nan
df.loc[df['contract_type'] == 'Two year', 'churn'] = df.loc[df['contract_type'] == 'Two year', 'churn'] * 0.3
```

---

## ✅ Этапы проекта

### Этап 1 — EDA (обязательно)

1. Полный EDA: форма, типы, пропуски, дубликаты
2. Анализ каждого признака с визуализацией
3. Анализ корреляций с таргетом
4. Сформулируй 5 гипотез о данных

### Этап 2 — Preprocessing и Feature Engineering

1. Очисти данные (пропуски, аномалии, несогласованные категории)
2. Создай минимум 10 новых признаков:
   - Отношения (charges_per_tenure_month, complaints_per_support_call и др.)
   - Агрегации по contract_type, internet_service
   - Бинарные флаги (is_high_value, is_long_term и др.)
3. Выбери финальный набор признаков

### Этап 3 — Baseline и сравнение моделей

1. Реализуй два baseline (random + majority class)
2. Обучи минимум 4 модели: LogReg, RandomForest, LightGBM, XGBoost
3. Сравни на 5-fold Stratified CV по ROC-AUC и Recall@top10%

### Этап 4 — Тюнинг лучшей модели

1. Оптимизируй гиперпараметры лучшей модели (Optuna, 50+ trials)
2. Проверь calibration вероятностей
3. Реализуй sklearn Pipeline (preprocessing + model)

### Этап 5 — Интерпретация

1. Feature importance (top-15)
2. SHAP values: global summary plot
3. Объясни 3 конкретных предсказания: почему модель решила именно так?
4. Найди сегменты клиентов с наибольшим риском оттока

### Этап 6 — Бизнес-оценка (обязательно!)

1. Переведи метрики в бизнес-язык:
   - При бюджете 10% базы — сколько % реального оттока поймаем?
   - Precision@10% (из топ-10% предсказаний, сколько реально уйдут?)
2. Рассчитай ROI: если удержание клиента стоит 500 руб, потеря = 2000 руб — выгодна ли модель?
3. Рекомендация: при каком threshold запускать кампанию?

### Этап 7 — API (бонус ⭐⭐⭐)

Создай FastAPI сервис:
```
POST /predict
{
  "tenure_months": 12,
  "monthly_charges": 65.0,
  ...
}
→ {"churn_probability": 0.73, "risk_level": "high", "recommendation": "offer_discount"}
```

Добавь три уровня риска: low (<30%), medium (30–60%), high (>60%).

---

## 📊 Критерии оценки

| Критерий | Вес | Что оценивается |
|----------|-----|-----------------|
| EDA | 15% | Глубина анализа, инсайты |
| Feature Engineering | 20% | Качество и осмысленность признаков |
| Моделирование | 20% | Правильная валидация, сравнение моделей |
| Метрики | 15% | Правильный выбор метрик для задачи |
| Интерпретация | 15% | Бизнес-объяснения, SHAP |
| Бизнес-оценка | 15% | Перевод метрик в реальные деньги |

---

## 🏆 Целевые показатели

- ROC-AUC ≥ 0.75 на test
- Recall@10% (recall среди топ-10% по вероятности) ≥ 0.40
- Весь код работает воспроизводимо (random_state=42 везде)
- Pipeline сохранён и загружается корректно

---

## 💡 Советы

- Начни с EDA — не спеши строить модели
- Записывай решения и их обоснование по ходу работы
- Baseline — обязателен, без него невозможно оценить прирост
- Test set трогай ТОЛЬКО один раз — в конце
- Финальный вывод пиши как будто объясняешь бизнес-заказчику (без ML жаргона)
