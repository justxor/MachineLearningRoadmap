# ✅ Разбор Задачи 05 — Обучение Моделей

**Задание:** [task-05-modeling.md](../practice-tasks/task-05-modeling.md)

---

## 🔑 Ключевые инсайты

1. **Accuracy = 75% при 25% дефолтов** — просто угадывание нулей. Метрика для этой задачи — **Recall** (FN дорог)
2. **Baseline обязателен** — без него нельзя оценить прирост модели
3. **Threshold = 0.5 по умолчанию** — почти никогда не оптимален. Всегда настраивай
4. **Learning curve** — главный инструмент диагностики overfitting/underfitting
5. **Test трогаем ОДИН раз** — всё что до этого — val, cv, train

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, roc_curve, classification_report)
import matplotlib.pyplot as plt

# Данные из условия
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'age':              np.random.randint(21, 70, n),
    'income':           np.abs(np.random.normal(50000, 30000, n)),
    'loan_amount':      np.abs(np.random.normal(15000, 10000, n)),
    'loan_term':        np.random.choice([12, 24, 36, 48, 60], n),
    'credit_score':     np.clip(np.random.normal(650, 100, n), 300, 850).astype(int).astype(float),
    'employment_years': np.random.randint(0, 30, n),
    'existing_loans':   np.random.randint(0, 5, n),
    'education':        np.random.choice(['High School','Bachelor','Master','PhD'], n, p=[0.3,0.4,0.2,0.1]),
    'home_ownership':   np.random.choice(['Own','Rent','Mortgage'], n, p=[0.3,0.4,0.3]),
    'default':          np.random.binomial(1, 0.25, n),
})
df.loc[np.random.choice(n, 100), 'credit_score'] = np.nan
df.loc[np.random.choice(n, 80), 'income'] = np.nan

X = df.drop(columns=['default'])
y = df['default']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
# 0.176 ≈ 0.15 / 0.85

# Preprocessing Pipeline
num_cols = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'employment_years', 'existing_loans']
cat_cols = ['education', 'home_ownership']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols),
])

# Baseline
y_pred_zero = np.zeros(len(y_val))
y_pred_rate = np.random.binomial(1, y_train.mean(), len(y_val))
print("Baseline (all zeros):", f"Acc={accuracy_score(y_val, y_pred_zero):.3f}",
      f"AUC={roc_auc_score(y_val, y_pred_zero):.3f}")
print("Baseline (random):", f"Acc={accuracy_score(y_val, y_pred_rate):.3f}",
      f"AUC={roc_auc_score(y_val, y_pred_rate):.3f}")

# Модели
models = {
    'LogisticRegression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'DecisionTree': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))]),
    'RandomForest': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]),
    'GradientBoosting': Pipeline([('prep', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))]),
}

results = {}
fig, ax = plt.subplots(figsize=(10, 6))

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = pipe.predict(X_val)

    metrics = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, y_proba),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'F1': f1_score(y_val, y_pred, zero_division=0),
    }
    results[name] = metrics

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={metrics['ROC-AUC']:.3f})")

ax.plot([0,1],[0,1],'k--', label='Random')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves'); ax.legend()
plt.show()

results_df = pd.DataFrame(results).T
print("\nСравнение моделей:")
print(results_df.round(3))

# Threshold optimization для лучшей модели (RandomForest)
best_pipe = models['RandomForest']
y_proba_val = best_pipe.predict_proba(X_val)[:, 1]

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)

# Найти threshold: Recall максимален при Precision >= 0.5
mask = precision[:-1] >= 0.50
if mask.any():
    best_t = thresholds[mask][np.argmax(recall[:-1][mask])]
    print(f"\nОптимальный threshold (Recall@Precision>=0.5): {best_t:.3f}")

# Финальная оценка на TEST (только 1 раз!)
y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= best_t).astype(int)
print(f"\nFINAL TEST RESULTS (threshold={best_t:.3f}):")
print(classification_report(y_test, y_pred_test))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
```

---

## 🐛 Типичные ошибки

1. **Смотреть только accuracy** — при 25% дефолтов предсказание "всегда 0" даёт 75%. Это не модель!

2. **Не делать baseline** — без него непонятно, хорошая ли модель с AUC=0.71.

3. **Threshold=0.5 по умолчанию** — в банковской задаче нужен более низкий threshold для повышения Recall.

4. **Трогать test несколько раз** — каждый раз подбираешь модель под test, и реальная оценка оказывается завышенной.

---

## 📌 Ключевые выводы

- **GradientBoosting/RandomForest** обычно лучше DT и LR на табличных данных
- **Бизнес-метрика:** максимизируем Recall при разумном Precision
- **Threshold:** снизить с 0.5 до ~0.35–0.40 для банковской задачи с ценой FN > FP
- **credit_score** — ожидаемо самый важный признак
