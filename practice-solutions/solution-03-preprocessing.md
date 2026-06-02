# ✅ Разбор Задачи 03 — Preprocessing

**Задание:** [task-03-preprocessing.md](../practice-tasks/task-03-preprocessing.md)

---

## 🔑 Ключевые инсайты

1. **SMOTE только на train** — применение до split приводит к утечке синтетических данных в val
2. **RobustScaler** лучше StandardScaler при наличии выбросов (monthly_income=150000)
3. **Флаги пропусков** (income_was_missing) могут быть предиктивны — пропуск дохода не случаен
4. **class_weight='balanced'** — самый простой способ борьбы с дисбалансом, часто лучший

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Данные из условия
np.random.seed(42)
n = 1470
df = pd.DataFrame({
    'age':                  np.random.randint(18, 60, n),
    'monthly_income':       np.abs(np.random.normal(6500, 4000, n)),
    'years_at_company':     np.random.randint(0, 40, n),
    'years_in_role':        np.random.randint(0, 18, n),
    'overtime':             np.random.choice(['Yes', 'No', 'yes', 'no', 'YES'], n),
    'job_satisfaction':     np.random.choice([1, 2, 3, 4, np.nan], n),
    'education':            np.random.choice(['Bachelor', 'Master', 'PhD', 'HighSchool', 'bachelor'], n),
    'department':           np.random.choice(['Sales', 'RD', 'HR'], n, p=[0.4, 0.4, 0.2]),
    'distance_from_home':   np.random.randint(1, 30, n),
    'work_life_balance':    np.random.choice([1, 2, 3, 4], n),
    'num_companies_worked': np.random.randint(0, 9, n),
    'attrition':            np.random.choice([1, 0], n, p=[0.16, 0.84]),
})
df.loc[np.random.choice(n, 80), 'monthly_income'] = np.nan
df.loc[np.random.choice(n, 60), 'job_satisfaction'] = np.nan
df.loc[np.random.choice(n, 30), 'education'] = np.nan
df.loc[5, 'monthly_income'] = 150000
df.loc[10, 'years_in_role'] = 99
df.loc[15, 'age'] = -5

# === ЧАСТЬ 1: Очистка ===

# 1.1 Стандартизация overtime
print("overtime уникальные:", df['overtime'].unique())
df['overtime'] = df['overtime'].str.lower().map({'yes': 1, 'no': 0})

# 1.2 Стандартизация education
df['education'] = df['education'].str.strip().str.title()
print("education после:", df['education'].value_counts())

# 1.3 Аномалии
print(f"age < 0: {(df['age'] < 0).sum()} строк")
print(f"years_in_role > 50: {(df['years_in_role'] > 50).sum()} строк")
print(f"monthly_income > 100000: {(df['monthly_income'] > 100000).sum()} строк")

# Решение: заменить на NaN (а не удалять строку — потеряем данные)
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['years_in_role'] > 50, 'years_in_role'] = np.nan
df.loc[df['monthly_income'] > 100000, 'monthly_income'] = np.nan

# === ЧАСТЬ 2: Обработка пропусков ===
print("\nПропуски:")
print((df.isnull().mean() * 100).round(2))

# ВАЖНО: добавить флаг ДО импутации
df['income_was_missing'] = df['monthly_income'].isnull().astype(int)
df['satisfaction_was_missing'] = df['job_satisfaction'].isnull().astype(int)

# Split до импутации!
X = df.drop(columns=['attrition'])
y = df['attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Числовые колонки
num_cols = ['age', 'monthly_income', 'years_at_company', 'years_in_role',
            'distance_from_home', 'work_life_balance', 'num_companies_worked', 'job_satisfaction']

# Медиана (fit ТОЛЬКО на train)
imp_median = SimpleImputer(strategy='median')
X_train[num_cols] = imp_median.fit_transform(X_train[num_cols])
X_test[num_cols] = imp_median.transform(X_test[num_cols])

# education — most_frequent
cat_imp = SimpleImputer(strategy='most_frequent')
X_train[['education']] = cat_imp.fit_transform(X_train[['education']])
X_test[['education']] = cat_imp.transform(X_test[['education']])

# === ЧАСТЬ 3: Кодирование и масштабирование ===

# department OHE
X_train = pd.get_dummies(X_train, columns=['department'], drop_first=False, dtype=float)
X_test = pd.get_dummies(X_test, columns=['department'], drop_first=False, dtype=float)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# education — Ordinal
edu_order = [['High School', 'Bachelor', 'Master', 'Phd']]
oe = OrdinalEncoder(categories=edu_order, handle_unknown='use_encoded_value', unknown_value=-1)
X_train['education'] = oe.fit_transform(X_train[['education']])
X_test['education'] = oe.transform(X_test[['education']])

# RobustScaler (устойчив к выбросам)
scaler = RobustScaler()
scale_cols = [c for c in num_cols if c in X_train.columns]
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# === ЧАСТЬ 4: Дисбаланс ===
print(f"\nКласс 1: {y_train.mean()*100:.1f}%")

# Подход 1: class_weight='balanced'
lr1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr1.fit(X_train, y_train)
y_proba1 = lr1.predict_proba(X_test)[:, 1]
y_pred1 = lr1.predict(X_test)

# Подход 2: SMOTE (ТОЛЬКО на train!)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
lr2 = LogisticRegression(max_iter=1000, random_state=42)
lr2.fit(X_resampled, y_resampled)
y_proba2 = lr2.predict_proba(X_test)[:, 1]
y_pred2 = lr2.predict(X_test)

print("\nСравнение подходов:")
for name, y_pred, y_proba in [('class_weight', y_pred1, y_proba1), ('SMOTE', y_pred2, y_proba2)]:
    print(f"  {name}: AUC={roc_auc_score(y_test, y_proba):.3f} | "
          f"F1={f1_score(y_test, y_pred):.3f} | "
          f"Recall={recall_score(y_test, y_pred):.3f} | "
          f"Precision={precision_score(y_test, y_pred):.3f}")
```

---

## 🐛 Типичные ошибки

1. **fit scaler на всём датасете** — scaler должен знать только о train. Иначе mean/std посчитаны с учётом test — утечка данных.

2. **SMOTE до split** — синтетические примеры из test попадут в train. Правильно: split → SMOTE только на X_train.

3. **Не добавлять флаг пропуска** — если доход отсутствует не случайно (например, у самозанятых), флаг сам по себе предиктивен.

4. **OneHotEncoder без reindex** — в test могут быть другие категории, нужно выравнивать колонки.

---

## 📌 Ключевые выводы

- **class_weight='balanced'** — самый быстрый и часто достаточный способ
- **RobustScaler** правильный выбор при выбросах (monthly_income=150000 искажает StandardScaler)
- **Порядок важен:** сначала split, потом fit all transformers только на train
- Accuracy=0.84 при дисбалансе 16/84 — это просто угадывание большинства. Смотри F1/Recall!
