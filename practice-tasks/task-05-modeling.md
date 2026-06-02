# 🤖 Задача 05 — Обучение Моделей

**Сложность:** ⭐⭐⭐ | **Время:** 120–180 минут
**Разбор:** [solution-05-modeling.md](../practice-solutions/solution-05-modeling.md)

---

## 📋 Контекст

Банк хочет предсказать, одобрить ли кредит заявителю. Задача — бинарная классификация. Бизнес-требование: минимизировать False Negatives (выдача кредита ненадёжному заёмщику) важнее, чем False Positives.

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'age':              np.random.randint(21, 70, n),
    'income':           np.abs(np.random.normal(50000, 30000, n)),
    'loan_amount':      np.abs(np.random.normal(15000, 10000, n)),
    'loan_term':        np.random.choice([12, 24, 36, 48, 60], n),
    'credit_score':     np.clip(np.random.normal(650, 100, n), 300, 850).astype(int),
    'employment_years': np.random.randint(0, 30, n),
    'existing_loans':   np.random.randint(0, 5, n),
    'education':        np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'home_ownership':   np.random.choice(['Own', 'Rent', 'Mortgage'], n, p=[0.3, 0.4, 0.3]),
    'default':          np.random.binomial(1, 0.25, n),
})

# Добавим пропуски
df.loc[np.random.choice(n, 100), 'credit_score'] = np.nan
df.loc[np.random.choice(n, 80), 'income'] = np.nan
```

---

## ✅ Задания

### Часть 1 — Baseline

**1.1.** Раздели данные на train (70%) / val (15%) / test (15%) со стратификацией по default.

**1.2.** Реализуй два baseline решения:
- Предсказывать всегда класс 0 (никто не дефолтит)
- Предсказывать по частоте класса (random с вероятностями из train)

Вычисли для каждого: Accuracy, ROC-AUC, Precision, Recall, F1.

**1.3.** Почему accuracy — плохая метрика для этой задачи? Какую метрику выбрать с учётом бизнес-требования?

---

### Часть 2 — Базовые модели

**2.1.** Обучи следующие модели на train, оцени на val:
- LogisticRegression
- DecisionTreeClassifier (max_depth=5)
- RandomForestClassifier (n_estimators=100)
- GradientBoostingClassifier

Для каждой: Accuracy, ROC-AUC, Precision, Recall, F1-score. Выведи таблицу сравнения.

**2.2.** Построй ROC кривые всех 4 моделей на одном графике.

**2.3.** Найди оптимальный threshold для лучшей модели, который максимизирует Recall при Precision >= 0.5.

---

### Часть 3 — Тюнинг гиперпараметров

**3.1.** Для RandomForestClassifier проведи GridSearchCV со следующими параметрами:
- n_estimators: [100, 200, 300]
- max_depth: [5, 10, 20, None]
- min_samples_leaf: [1, 5, 10]
- scoring='roc_auc', cv=5

Сколько вариантов перебирается? Сколько времени ушло?

**3.2.** Для LightGBM (установи если нет: pip install lightgbm) проведи Optuna оптимизацию (50 trials). Сравни с GridSearch: качество vs время.

**3.3.** Построй кривые обучения (learning curves) для лучшей модели. Есть ли overfitting или underfitting?

---

### Часть 4 — Интерпретация и деплой модели

**4.1.** Для лучшей модели:
- Выведи feature importance (top-10)
- Посчитай SHAP values для 5 случайных примеров
- Объясни одно конкретное предсказание: почему модель одобрила/отказала?

**4.2.** Оцени финальную модель на TEST (только один раз!). Сравни с val метриками — есть ли сильное расхождение?

**4.3. (Бонус)** Откалибруй вероятности модели. Построй calibration plot. Улучшилось ли качество?

---

## 📝 Требования к ответу

- Таблица сравнения всех моделей
- Объяснение выбора финальной модели
- Объяснение выбора метрики для этой задачи
- Финальная оценка на test set
- Рекомендация: при каком threshold одобрять кредит?

---

## 💡 Подсказки

<details>
<summary>Подсказка к 1.2 (baseline)</summary>

```python
# Всегда 0
y_pred_zero = np.zeros(len(y_val))
# Случайный с вероятностями
pos_rate = y_train.mean()
y_pred_random = np.random.binomial(1, pos_rate, len(y_val))
```
</details>

<details>
<summary>Подсказка к 2.3 (threshold)</summary>

```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
# Найти threshold где precision >= 0.5 и recall максимален
mask = precision[:-1] >= 0.5
best_threshold = thresholds[mask][np.argmax(recall[:-1][mask])]
```
</details>
