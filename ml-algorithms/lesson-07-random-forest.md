# Урок 07: Random Forest — ансамбль деревьев

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 06](lesson-06-knn.md) | [Урок 08 →](lesson-08-gradient-boosting-theory.md)

---

## 🎯 Цели урока

- Понять Bagging как метод снижения дисперсии
- Разобраться как случайный выбор признаков снижает корреляцию деревьев
- Освоить OOB evaluation и Feature Importance
- Научиться подбирать гиперпараметры Random Forest

---

## 📐 Теория

### Bagging (Bootstrap Aggregating)

Идея: N слабых моделей (high variance) → 1 сильная (low variance)

**Bootstrap:** выборка N примеров с возвращением → ~63.2% уникальных примеров в каждой выборке.

**Aggregating:**
- Классификация: majority vote
- Регрессия: среднее предсказаний

### Random Forest = Bagging деревьев + случайные признаки

При каждом сплите выбираем случайное подмножество из max_features признаков.

Зачем? Снижает корреляцию деревьев → лучше дисперсия ансамбля.

Типичные значения max_features:
- Классификация: √d (например, при d=100 → 10 признаков)
- Регрессия: d/3

### OOB Error (Out-of-Bag)

~36.8% примеров не попали в bootstrap выборку → можно использовать для валидации «бесплатно».

### Feature Importance

```
Importance(feature j) = Σ (деревья) Σ (узлы с j) [снижение Gini × долю примеров]
```

Нормируется на сумму → интерпретируется как % вклада.

---

## 💻 Реализация

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение Random Forest
rf = RandomForestClassifier(
    n_estimators=100,    # число деревьев
    max_depth=10,        # максимальная глубина
    max_features='sqrt', # √d признаков при каждом сплите
    min_samples_split=5, # минимум примеров для разбиения
    oob_score=True,      # OOB evaluation
    random_state=42,
    n_jobs=-1            # параллельное обучение
)
rf.fit(X_train, y_train)

print(f"Test accuracy:  {rf.score(X_test, y_test):.4f}")
print(f"OOB accuracy:   {rf.oob_score_:.4f}")  # "бесплатная" валидация

# Feature Importance
feature_names = data.feature_names
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Топ-15 признаков
ax = axes[0]
top_n = 15
ax.bar(range(top_n), importances[indices[:top_n]])
ax.set_xticks(range(top_n))
ax.set_xticklabels([feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
ax.set_title('Feature Importances (Random Forest)')
ax.set_ylabel('Importance')

# Кривая обучения: n_estimators vs OOB error
ax = axes[1]
oob_errors = []
estimator_range = range(1, 201, 10)
for n in estimator_range:
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf_n.fit(X_train, y_train)
    oob_errors.append(1 - rf_n.oob_score_)

ax.plot(list(estimator_range), oob_errors)
ax.set_xlabel('Number of trees'); ax.set_ylabel('OOB Error Rate')
ax.set_title('OOB Error vs Number of Trees')

plt.tight_layout(); plt.show()


# Permutation Importance (альтернативный метод)
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
print("\nTop 5 по Permutation Importance:")
for idx in result.importances_mean.argsort()[::-1][:5]:
    print(f"  {feature_names[idx]}: {result.importances_mean[idx]:.4f} ± {result.importances_std[idx]:.4f}")
```

---

## 🔑 Ключевые выводы

1. **Bagging** снижает дисперсию: среднее N моделей более стабильно чем одна
2. **Случайные признаки** делают деревья менее коррелированными → лучший ансамбль
3. **OOB error** = "бесплатная" оценка без отдельного validation set
4. **Feature importance** через Gini bias к высококардинальным признакам → используйте Permutation Importance
5. **100-500 деревьев** достаточно: OOB error стабилизируется

---

## ⚡ Практические задания

1. Сравните Random Forest с одиночным деревом: bias, variance, accuracy
2. Постройте кривую OOB error vs n_estimators. Где наступает насыщение?
3. Сравните Gini importance vs Permutation importance. На каком датасете они расходятся?
4. Подберите гиперпараметры через GridSearchCV: max_depth, max_features, n_estimators

---

**[← Урок 06: KNN](lesson-06-knn.md)** | **[Урок 08: Gradient Boosting →](lesson-08-gradient-boosting-theory.md)**
