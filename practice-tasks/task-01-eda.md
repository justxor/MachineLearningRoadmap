# 🔍 Задача 01 — EDA: Анализ Страховых Выплат

**Сложность:** ⭐⭐ | **Время:** 60–90 минут
**Разбор:** [solution-01-eda.md](../practice-solutions/solution-01-eda.md)

---

## 📋 Контекст

Ты аналитик в страховой компании. Нужно исследовать датасет с клиентами перед построением модели предсказания страховых выплат.

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1338

df = pd.DataFrame({
    'age':      np.random.randint(18, 65, n),
    'sex':      np.random.choice(['male', 'female'], n),
    'bmi':      np.round(np.random.normal(30, 6, n), 1),
    'children': np.random.randint(0, 5, n),
    'smoker':   np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
    'region':   np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n),
    'charges':  np.abs(np.random.normal(13000, 12000, n)),
})
df.loc[np.random.choice(n, 50), 'bmi'] = np.nan
df.loc[np.random.choice(n, 30), 'age'] = np.nan
df.loc[5, 'bmi'] = 85.0       # выброс
df.loc[10, 'charges'] = -500  # аномалия
```

---

## ✅ Задания

### Часть 1 — Базовое исследование

1.1. Размер, типы колонок, пропуски, дубликаты.

1.2. Для числовых колонок: mean, median, std, min, max, Q1, Q3, skewness, kurtosis. Какие имеют |skew| > 1?

1.3. Для категориальных: уникальные значения, частоты, дисбаланс.

1.4. Найди аномалии: отрицательный charges, bmi > 60, age < 0 или > 100.

---

### Часть 2 — Распределения

2.1. Гистограмма + KDE для каждой числовой колонки. Опиши форму каждого распределения.

2.2. Boxplot charges по: smoker, sex, region. Какой фактор влияет сильнее?

2.3. Scatter plot bmi vs charges, раскрашенный по smoker. Что видно?

---

### Часть 3 — Корреляции

3.1. Матрица Pearson корреляций. Топ-3 признака по корреляции с charges.

3.2. Повтори для Spearman. Отличия?

3.3. Бинаризируй smoker (0/1) и посчитай корреляцию с charges.

---

### Часть 4 — Статистические тесты (бонус ⭐)

4.1. Тест Шапиро-Уилка на нормальность charges.

4.2. t-тест: отличаются ли charges у курящих и некурящих? p-value и вывод.

4.3. Создай bmi_category (underweight/normal/overweight/obese), проанализируй charges по ней.

---

## 📝 Требования к ответу

- Краткий вывод после каждой части
- Все графики с подписями и заголовками
- Минимум 5 инсайтов о данных
- Список проблем, которые нужно решить до моделирования

---

## 💡 Подсказки

<details>
<summary>Подсказка к 1.1</summary>

`df.info()`, `df.isnull().sum()`, `df.duplicated().sum()`
</details>

<details>
<summary>Подсказка к 4.2</summary>

`from scipy.stats import ttest_ind`
`ttest_ind(df[df.smoker=='yes'].charges.dropna(), df[df.smoker=='no'].charges.dropna())`
</details>
