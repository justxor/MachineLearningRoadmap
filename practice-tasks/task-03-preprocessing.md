# ⚙️ Задача 03 — Предобработка Данных

**Сложность:** ⭐⭐ | **Время:** 60–90 минут
**Разбор:** [solution-03-preprocessing.md](../practice-solutions/solution-03-preprocessing.md)

---

## 📋 Контекст

HR-аналитик в крупной компании. Нужно предсказать, уволится ли сотрудник (churn). Данные грязные: пропуски, выбросы, смешанные типы, сильный дисбаланс классов (только 16% увольняются).

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

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

# Добавляем проблемы
df.loc[np.random.choice(n, 80), 'monthly_income'] = np.nan
df.loc[np.random.choice(n, 60), 'job_satisfaction'] = np.nan
df.loc[np.random.choice(n, 30), 'education'] = np.nan
df.loc[5, 'monthly_income'] = 150000   # выброс
df.loc[10, 'years_in_role'] = 99       # аномалия
df.loc[15, 'age'] = -5                 # ошибка
```

---

## ✅ Задания

### Часть 1 — Очистка данных

**1.1.** Найди и исправь проблемы в overtime: значения 'Yes', 'yes', 'YES' — одно и то же. Приведи к единому виду 0/1.

**1.2.** Найди и исправь дубликаты в education ('Bachelor' и 'bachelor'). Приведи к единому регистру.

**1.3.** Найди строки с аномальными значениями: age < 0, years_in_role > 50, monthly_income > 100000. Реши: удалить или заменить? Обоснуй решение.

---

### Часть 2 — Обработка пропусков

**2.1.** Посчитай процент пропусков в каждой колонке. Нарисуй визуализацию пропусков.

**2.2.** Заполни пропуски в monthly_income тремя способами:
- Медиана по всему датасету
- Медиана в разрезе department (группировка)
- KNN импутация (k=5)

Сравни результаты: посчитай RMSE между тремя вариантами на строках где были пропуски (создай маску заранее).

**2.3.** Для job_satisfaction — категориальный порядковый признак. Какой метод импутации лучше? Реализуй.

**2.4.** Добавь бинарный флаг income_was_missing перед импутацией. Почему это может улучшить модель?

---

### Часть 3 — Кодирование и масштабирование

**3.1.** Закодируй department:
- One-Hot Encoding
- Ordinal Encoding (предположи порядок: HR < Sales < RD)
Какой метод лучше для RandomForest? Для LogisticRegression?

**3.2.** Масштабируй числовые признаки. Выбери между StandardScaler и RobustScaler на основе анализа выбросов. Обоснуй выбор.

**3.3.** ВАЖНО: Реализуй правильный pipeline — fit scaler ТОЛЬКО на train, apply на train+test.
Покажи, что результаты на test изменятся, если сделать fit на всём датасете (data leakage).

---

### Часть 4 — Дисбаланс классов (бонус ⭐⭐)

**4.1.** Посчитай соотношение классов. Обучи LogisticRegression без балансировки и оцени по accuracy и ROC-AUC. Почему accuracy вводит в заблуждение?

**4.2.** Попробуй три подхода к дисбалансу:
- class_weight='balanced' в модели
- SMOTE (oversample minority)
- Random undersampling

Для каждого: выведи ROC-AUC, Precision, Recall, F1 на val. Какой подход лучше?

**4.3.** Объясни: почему нельзя применять SMOTE до разбиения на train/val?

---

## 📝 Требования к ответу

- Код с комментариями для каждого шага
- Обоснование каждого решения (почему именно этот метод)
- Финальный размер датасета после очистки
- Сравнение метрик с балансировкой и без
