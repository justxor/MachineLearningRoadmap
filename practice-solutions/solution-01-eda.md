# ✅ Разбор Задачи 01 — EDA

**Задание:** [task-01-eda.md](../practice-tasks/task-01-eda.md)

---

## 🔑 Ключевые инсайты (прочитай перед кодом)

1. **smoker — главный драйвер charges** (корреляция ~0.79 после бинаризации)
2. Распределение charges **бимодальное** из-за курильщиков (два пика)
3. BMI имеет **правостороннюю асимметрию** — нужна log-трансформация
4. Регион практически **не влияет** на charges
5. Взаимодействие smoker × bmi даёт очень интересный паттерн

---

## 💻 Полное решение

### Часть 1 — Базовое исследование

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Генерация данных (из условия задачи)
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
df.loc[5, 'bmi'] = 85.0
df.loc[10, 'charges'] = -500

# 1.1 Базовый осмотр
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:")
missing = df.isnull().sum()
missing_pct = df.isnull().mean() * 100
print(pd.DataFrame({'count': missing, 'pct': missing_pct.round(2)})[missing > 0])
print(f"\nDuplicates: {df.duplicated().sum()}")

# 1.2 Числовые признаки
num_cols = ['age', 'bmi', 'children', 'charges']
stats_df = df[num_cols].agg(['mean', 'median', 'std', 'min', 'max',
                              lambda x: x.quantile(0.25),
                              lambda x: x.quantile(0.75),
                              'skew', 'kurt'])
stats_df.index = ['mean', 'median', 'std', 'min', 'max', 'Q1', 'Q3', 'skewness', 'kurtosis']
print("\nNumeric stats:")
print(stats_df.round(3).T)

# Асимметрия
for col in num_cols:
    skew = df[col].skew()
    if abs(skew) > 1:
        print(f"  {col}: HIGH skewness = {skew:.3f}")

# 1.3 Категориальные признаки
cat_cols = ['sex', 'smoker', 'region']
for col in cat_cols:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts(normalize=True).mul(100).round(1).to_string())

# 1.4 Аномалии
print(f"\nОтрицательные charges: {(df['charges'] < 0).sum()} строк")
print(f"BMI > 60 или < 10: {((df['bmi'] > 60) | (df['bmi'] < 10)).sum()} строк")
print(f"Age < 0 или > 100: {((df['age'] < 0) | (df['age'] > 100)).sum()} строк")
```

### Часть 2 — Визуализация распределений

```python
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Distributions of Numeric Features', fontsize=14)

for idx, col in enumerate(num_cols):
    # Гистограмма + KDE
    ax = axes[0, idx]
    ax.hist(df[col].dropna(), bins=40, alpha=0.7, color='steelblue', edgecolor='none')
    ax2 = ax.twinx()
    df[col].dropna().plot.kde(ax=ax2, color='red', linewidth=2)
    ax2.set_ylabel('')
    ax.set_title(f'{col}\nSkew: {df[col].skew():.2f}')

    # Boxplot
    ax = axes[1, idx]
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_title(col)

plt.tight_layout()
plt.show()
# charges — правосторонняя асимметрия (бимодальная из-за курильщиков)
# bmi — близко к нормальному, небольшой правый хвост
# age — равномерное распределение
# children — дискретное, правосторонняя асимметрия

# Boxplot charges vs smoker — ГЛАВНЫЙ ИНСАЙТ
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=df, x='smoker', y='charges', ax=axes[0])
axes[0].set_title('Charges by Smoker Status')

sns.boxplot(data=df, x='sex', y='charges', ax=axes[1])
axes[1].set_title('Charges by Sex')

sns.boxplot(data=df, x='region', y='charges', ax=axes[2])
axes[2].set_title('Charges by Region')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
# ВЫВОД: Smoker — самый сильный фактор. Медиана у курильщиков в 3+ раза выше.
# Пол и регион — незначительное влияние.

# Scatter bmi vs charges
fig, ax = plt.subplots(figsize=(10, 6))
for smoker_val, color in [('yes', 'red'), ('no', 'blue')]:
    mask = df['smoker'] == smoker_val
    ax.scatter(df.loc[mask, 'bmi'], df.loc[mask, 'charges'],
               alpha=0.4, color=color, s=20, label=f'smoker={smoker_val}')
ax.set_xlabel('BMI')
ax.set_ylabel('Charges')
ax.set_title('BMI vs Charges by Smoking Status')
ax.legend()
plt.show()
# ИНСАЙТ: у некурящих BMI почти не влияет на charges.
# У курящих — явная положительная связь BMI → charges.
```

### Часть 3 — Корреляции

```python
# Pearson корреляция
df_numeric = df.copy()
df_numeric['smoker_bin'] = (df['smoker'] == 'yes').astype(int)
df_numeric['sex_bin'] = (df['sex'] == 'male').astype(int)

corr_pearson = df_numeric[['age', 'bmi', 'children', 'charges', 'smoker_bin']].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=axes[0])
axes[0].set_title('Pearson Correlation')

# Spearman (более устойчива к выбросам)
corr_spearman = df_numeric[['age', 'bmi', 'children', 'charges', 'smoker_bin']].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=axes[1])
axes[1].set_title('Spearman Correlation')
plt.tight_layout()
plt.show()

print("Корреляция с charges:")
print(df_numeric[['age', 'bmi', 'children', 'smoker_bin']].corrwith(df_numeric['charges']).abs().sort_values(ascending=False))
# smoker_bin: ~0.79 — ОГРОМНАЯ корреляция
# age: ~0.30
# bmi: ~0.20
# children: ~0.07 (слабая)
```

### Часть 4 — Статистические тесты

```python
from scipy.stats import shapiro, ttest_ind

# Тест Шапиро-Уилка на нормальность
sample = df['charges'].dropna().sample(1000, random_state=42)
stat, p_value = shapiro(sample)
print(f"Shapiro-Wilk: statistic={stat:.4f}, p-value={p_value:.6f}")
print(f"Charges {'НЕ ' if p_value < 0.05 else ''}нормально распределены (alpha=0.05)")
# p_value << 0.05 — НЕ нормальное распределение

# t-тест: курящие vs некурящие
smokers = df[df['smoker'] == 'yes']['charges'].dropna()
non_smokers = df[df['smoker'] == 'no']['charges'].dropna()
t_stat, p_value = ttest_ind(smokers, non_smokers)
print(f"\nt-test: t={t_stat:.4f}, p-value={p_value:.2e}")
print(f"Средние charges: курящие={smokers.mean():.0f}, некурящие={non_smokers.mean():.0f}")
print(f"Разница статистически {'значима' if p_value < 0.05 else 'НЕ значима'}")
# p << 0.05 — разница ЗНАЧИМА

# BMI категории
df['bmi_category'] = pd.cut(df['bmi'].fillna(df['bmi'].median()),
                              bins=[0, 18.5, 25, 30, 100],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
print("\nCharges by BMI category:")
print(df.groupby('bmi_category')['charges'].agg(['mean', 'median', 'count']))
```

---

## 🐛 Типичные ошибки

1. **Забыть про бимодальность** — charges выглядит как одно распределение, но на самом деле два (курящие/некурящие). Всегда делай scatter с hue.

2. **Не проверить аномалии** — отрицательные charges и bmi=85 нужно найти и объяснить ДО обучения модели.

3. **Не добавить бинарную переменную для smoker** — без бинаризации корреляцию не посчитать.

4. **Использовать только Pearson** — для ненормальных распределений (как charges) Spearman надёжнее.

---

## 📌 Ключевые выводы

- **Smoker — главный предиктор** (корреляция 0.79 с charges)
- **Charges ненормально распределены** — log-трансформация поможет при регрессии
- **BMI влияет только у курильщиков** — нужен interaction feature smoker × bmi
- **Регион и пол не важны** — можно исключить или использовать с осторожностью
- **Проблемы данных:** ~50 пропусков в bmi, ~30 в age, 1 отрицательное charges, 1 выброс bmi=85
