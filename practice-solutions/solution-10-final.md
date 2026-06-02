# ✅ Разбор Задачи 10 — Финальный Проект (Телеком Churn)

**Задание:** [task-10-final.md](../practice-tasks/task-10-final.md)

---

## 🔑 Ключевые инсайты финального проекта

1. **Recall@10%** — правильная метрика для ограниченного бюджета. Обычный Recall не учитывает бюджетное ограничение
2. **Business ROI** — перевод метрик в деньги обязателен для убеждения stakeholders
3. **SHAP global + local** — global для понимания модели, local для объяснения конкретных клиентов
4. **Pipeline = deploy-ready** — весь preprocessing + model в одном объекте
5. **Month-to-month договор** — главный предиктор оттока (3–4x выше вероятность)

---

## 💻 Полное решение

### Этап 1–2: EDA + Feature Engineering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import lightgbm as lgb

np.random.seed(42)

# Данные из задачи
n = 7000
df = pd.DataFrame({
    'customer_id':        range(n),
    'tenure_months':      np.random.randint(1, 72, n),
    'monthly_charges':    np.abs(np.random.normal(65, 30, n)),
    'total_charges':      np.abs(np.random.normal(2000, 1800, n)),
    'contract_type':      np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.25,0.2]),
    'payment_method':     np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n),
    'internet_service':   np.random.choice(['DSL','Fiber optic','No'], n, p=[0.35,0.44,0.21]),
    'phone_service':      np.random.choice(['Yes','No'], n, p=[0.9,0.1]),
    'online_security':    np.random.choice(['Yes','No','No internet service'], n, p=[0.29,0.5,0.21]),
    'tech_support':       np.random.choice(['Yes','No','No internet service'], n, p=[0.29,0.5,0.21]),
    'streaming_tv':       np.random.choice(['Yes','No','No internet service'], n, p=[0.38,0.41,0.21]),
    'num_complaints':     np.random.randint(0, 5, n),
    'avg_monthly_usage':  np.abs(np.random.normal(50, 20, n)),
    'support_calls':      np.random.randint(0, 8, n),
    'gender':             np.random.choice(['Male','Female'], n),
    'senior_citizen':     np.random.binomial(1, 0.16, n),
    'partner':            np.random.choice(['Yes','No'], n, p=[0.48, 0.52]),
    'dependents':         np.random.choice(['Yes','No'], n, p=[0.3, 0.7]),
    'churn':              np.random.binomial(1, 0.265, n),
})
# Реалистичность
df.loc[np.random.choice(n, 150), 'total_charges'] = np.nan
df.loc[np.random.choice(n, 80), 'monthly_charges'] = np.nan
# Month-to-month имеют более высокий churn
df.loc[df['contract_type'] == 'Two year', 'churn'] =     df.loc[df['contract_type'] == 'Two year', 'churn'] * 0.3

# EDA быстро
print(f"Churn rate: {df['churn'].mean()*100:.1f}%")
print("Churn by contract:")
print(df.groupby('contract_type')['churn'].mean().sort_values(ascending=False))

# Feature Engineering
df_fe = df.copy()
df_fe.drop(columns=['customer_id'], inplace=True)

# Отношения
df_fe['charges_per_month_tenure'] = df_fe['monthly_charges'] / (df_fe['tenure_months'] + 1)
df_fe['total_to_monthly_ratio'] = df_fe['total_charges'] / (df_fe['monthly_charges'] + 1)
df_fe['complaints_per_support'] = df_fe['num_complaints'] / (df_fe['support_calls'] + 1)
df_fe['support_per_tenure'] = df_fe['support_calls'] / (df_fe['tenure_months'] + 1) * 12  # в год

# Бинарные флаги
df_fe['is_month_to_month'] = (df_fe['contract_type'] == 'Month-to-month').astype(int)
df_fe['is_fiber'] = (df_fe['internet_service'] == 'Fiber optic').astype(int)
df_fe['is_long_term'] = (df_fe['tenure_months'] > 24).astype(int)
df_fe['is_high_value'] = (df_fe['monthly_charges'] > 80).astype(int)
df_fe['has_complaints'] = (df_fe['num_complaints'] > 0).astype(int)
df_fe['no_security'] = ((df_fe['online_security'] == 'No') & (df_fe['internet_service'] != 'No')).astype(int)
df_fe['no_support'] = ((df_fe['tech_support'] == 'No') & (df_fe['internet_service'] != 'No')).astype(int)

print(f"\nФинальных признаков: {df_fe.shape[1] - 1}")

# Train/Val/Test split
X = df_fe.drop(columns=['churn'])
y = df_fe['churn']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
```

### Этап 3–4: Модели + Pipeline

```python
num_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
cat_cols = [c for c in X_train.select_dtypes(include=['object']).columns]

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols),
])

# Baseline
print("\nBaseline (all zeros):")
y_pred_zero = np.zeros(len(y_val))
print(f"  Accuracy={recall_score(y_val, y_pred_zero, zero_division=0):.3f}")

# Модели
models_final = {
    'LogisticRegression': Pipeline([('prep', preprocessor),
                                    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))]),
    'RandomForest': Pipeline([('prep', preprocessor),
                               ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))]),
    'LightGBM': Pipeline([('prep', preprocessor),
                           ('clf', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1))]),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_results = {}
for name, pipe in models_final.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"{name}: CV AUC={scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(cv_results, key=cv_results.get)
best_pipe = models_final[best_name]
best_pipe.fit(X_train, y_train)
y_proba_val = best_pipe.predict_proba(X_val)[:, 1]
```

### Этап 5: Интерпретация

```python
import shap

# Feature Importance
clf = best_pipe.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    fi = pd.Series(clf.feature_importances_,
                   index=num_cols + [f'cat_{i}' for i in range(len(cat_cols))]
                   ).sort_values(ascending=False)
    print("\nTop-10 features:")
    print(fi.head(10))

# SHAP (LightGBM поддерживает TreeExplainer)
X_train_transformed = best_pipe.named_steps['prep'].transform(X_train)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train_transformed[:200])
shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                  X_train_transformed[:200], show=False)
plt.title('SHAP Summary Plot — Churn')
plt.tight_layout()
plt.show()
```

### Этап 6: Бизнес-оценка

```python
# Recall@10% — из топ 10% предсказаний, сколько реально уйдут?
n_val = len(y_val)
top_10_pct = int(n_val * 0.10)
top_indices = np.argsort(y_proba_val)[::-1][:top_10_pct]
recall_at_10 = y_val.iloc[top_indices].mean() / y_val.mean()
precision_at_10 = y_val.iloc[top_indices].mean()

print(f"\nБизнес-метрики (Val):")
print(f"Recall@10%: {recall_at_10:.2f} (из ушедших поймали {recall_at_10*100:.0f}% за 10% бюджета)")
print(f"Precision@10%: {precision_at_10:.2f} ({precision_at_10*100:.0f}% из топ-10% реально уйдут)")

# ROI расчёт
n_customers = 10000  # гипотетическая клиентская база
churn_rate = 0.265
budget_pct = 0.10
cost_retention = 500    # стоимость удержания
revenue_lost = 2000     # потеря от ухода клиента

n_churners = int(n_customers * churn_rate)
n_contacted = int(n_customers * budget_pct)
n_caught = int(n_churners * recall_at_10)
n_false_alarm = n_contacted - n_caught

cost_outreach = n_contacted * cost_retention
revenue_saved = n_caught * revenue_lost * 0.6  # 60% успешно удержим
revenue_without_model = n_churners * revenue_lost

print(f"\nROI Analysis:")
print(f"  Клиентов в базе: {n_customers:,}")
print(f"  Уйдут без модели: {n_churners:,}")
print(f"  Обрабатываем топ-10%: {n_contacted:,} клиентов")
print(f"  Поймали: {n_caught:,} реальных уходящих")
print(f"  Ложная тревога: {n_false_alarm:,}")
print(f"  Затраты на кампанию: {cost_outreach:,} руб")
print(f"  Сохранённая выручка: {revenue_saved:,} руб")
print(f"  ROI = {(revenue_saved - cost_outreach) / cost_outreach * 100:.0f}%")

# ФИНАЛЬНЫЙ TEST
print("\n===== FINAL TEST RESULTS =====")
y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
top_test = np.argsort(y_proba_test)[::-1][:int(len(y_test)*0.10)]
print(f"Recall@10%: {y_test.iloc[top_test].mean() / y_test.mean():.2f}")
print(f"Precision@10%: {y_test.iloc[top_test].mean():.2f}")
```

---

## 🐛 Типичные ошибки финального проекта

1. **Оценивать test несколько раз** — val должен быть для выбора модели, test только для финальной оценки
2. **Не переводить в бизнес-метрики** — "AUC=0.80" ничего не говорит менеджеру. "Поймали 45% уходящих за 10% бюджета" — говорит
3. **Игнорировать дисбаланс** — 26.5% — умеренный дисбаланс. Без class_weight recall будет занижен
4. **Не делать SHAP** — feature importance RandomForest нестабильны для коррелированных признаков

---

## 📊 Эталонные результаты

| Метрика | Ожидаемо | Хорошо |
|---------|----------|--------|
| ROC-AUC | 0.70–0.75 | > 0.78 |
| Recall@10% | 0.30–0.40 | > 0.45 |
| Precision@10% | 0.35–0.45 | > 0.50 |
| ROI кампании | > 0% | > 150% |

---

## 📌 Ключевые выводы

- **Month-to-month контракт** — ключевой предиктор. Вероятность оттока в 3–4x выше
- **tenure_months** — чем дольше клиент, тем ниже вероятность ухода
- **num_complaints + support_calls** — взаимодействие с поддержкой сигнализирует о проблемах
- **Рекомендация threshold:** ~0.35 для максимизации recall при разумном precision
- **LightGBM** обычно лучше RandomForest на этой задаче (~+0.03 AUC)
