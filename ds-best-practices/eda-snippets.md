# 🔍 EDA — Разведочный Анализ Данных

Готовые сниппеты для быстрого и глубокого исследования данных.

---

## 1. Базовый осмотр датасета

```python
import pandas as pd
import numpy as np

def full_eda_report(df: pd.DataFrame) -> None:
    """Полный EDA отчёт в одной функции"""
    print("=" * 60)
    print(f"DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicates: {df.duplicated().sum():,} ({df.duplicated().mean()*100:.2f}%)")

    print("\n--- DATA TYPES ---")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
        print(missing_df.to_string())
    else:
        print("  No missing values!")

    print("\n--- NUMERIC COLUMNS ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  {len(num_cols)} columns: {num_cols}")

    print("\n--- CATEGORICAL COLUMNS ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"  {len(cat_cols)} columns: {cat_cols}")
    for col in cat_cols[:5]:  # первые 5
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique values")
        if n_unique <= 10:
            print(f"    {df[col].value_counts().to_dict()}")

# Использование:
# full_eda_report(df)
```

---

## 2. Анализ числовых признаков

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_numeric(df: pd.DataFrame, cols: list = None, target: str = None):
    """Детальный анализ числовых колонок"""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in cols:
            cols.remove(target)

    summary = []
    for col in cols:
        series = df[col].dropna()
        stat = {
            'column': col,
            'count': len(series),
            'missing': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().mean() * 100,
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'zeros_pct': (series == 0).mean() * 100,
            'negatives_pct': (series < 0).mean() * 100,
        }
        # Количество выбросов (метод IQR)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
        stat['outliers_count'] = outliers
        stat['outliers_pct'] = outliers / len(series) * 100

        # Тест на нормальность (для малых выборок)
        if len(series) <= 5000:
            _, pvalue = stats.shapiro(series.sample(min(len(series), 1000), random_state=42))
            stat['shapiro_p'] = pvalue
            stat['is_normal'] = pvalue > 0.05

        summary.append(stat)

    return pd.DataFrame(summary).set_index('column')

# Использование:
# report = analyze_numeric(df, target='price')
# print(report.to_string())
```

---

## 3. Визуализация распределений

```python
def plot_distributions(df: pd.DataFrame, cols: list = None, n_cols: int = 3):
    """Гистограммы + boxplot для числовых признаков"""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6 * n_cols, 4 * n_rows * 2))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

    for idx, col in enumerate(cols):
        row_hist = (idx // n_cols) * 2
        row_box = row_hist + 1
        col_idx = idx % n_cols

        data = df[col].dropna()

        # Гистограмма с KDE
        ax1 = axes[row_hist, col_idx] if n_rows > 1 else axes[0, col_idx]
        ax1.hist(data, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax1_twin = ax1.twinx()
        data.plot.kde(ax=ax1_twin, color='red', linewidth=2)
        ax1_twin.set_ylabel('')
        ax1.set_title(f'{col}\nSkew: {data.skew():.2f}', fontweight='bold')
        ax1.set_xlabel('')

        # Boxplot
        ax2 = axes[row_box, col_idx] if n_rows > 1 else axes[1, col_idx]
        ax2.boxplot(data, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        ax2.set_xlabel(col)

    # Скрываем лишние оси
    for idx in range(len(cols), n_rows * n_cols):
        for r in [0, 1]:
            axes[(idx // n_cols) * 2 + r, idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_distributions(df, cols=['age', 'salary', 'experience'])
```

---

## 4. Матрица корреляций

```python
def plot_correlation_matrix(df: pd.DataFrame, target: str = None,
                             method: str = 'pearson', figsize=(14, 12)):
    """Красивая матрица корреляций"""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)

    # Маска для верхнего треугольника
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # Топ корреляций с таргетом
    if target and target in corr:
        print(f"\nТоп корреляции с '{target}':")
        target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
        for feat, val in target_corr.head(10).items():
            direction = "+" if corr[target][feat] > 0 else "-"
            print(f"  {feat}: {direction}{val:.3f}")

    return corr

# corr_matrix = plot_correlation_matrix(df, target='price')
```

---

## 5. Анализ категориальных признаков

```python
def analyze_categorical(df: pd.DataFrame, target: str = None, max_categories: int = 20):
    """Анализ категориальных переменных с target encoding preview"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"\n{'='*50}")
        print(f"Column: {col} | Unique: {n_unique} | Missing: {df[col].isnull().sum()}")

        if n_unique > max_categories:
            print(f"  Too many categories ({n_unique}), showing top 10:")
            print(df[col].value_counts().head(10).to_string())
        else:
            print(df[col].value_counts(normalize=True).mul(100).round(1).to_string())

        if target and target in df.columns:
            if df[target].dtype in [np.float64, np.int64]:
                # Числовой таргет — показываем среднее
                group_stats = df.groupby(col)[target].agg(['mean', 'median', 'count'])
                group_stats = group_stats.sort_values('mean', ascending=False)
                print(f"\n  Target '{target}' by {col}:")
                print(group_stats.to_string())

# analyze_categorical(df, target='price')
```

---

## 6. Анализ выбросов

```python
def detect_outliers(df: pd.DataFrame, cols: list = None, method: str = 'iqr'):
    """Обнаружение выбросов — IQR или Z-score"""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = {}
    for col in cols:
        series = df[col].dropna()
        if method == 'iqr':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > 3
            lower = series.mean() - 3 * series.std()
            upper = series.mean() + 3 * series.std()
        elif method == 'percentile':
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            outlier_mask = (df[col] < lower) | (df[col] > upper)

        results[col] = {
            'n_outliers': outlier_mask.sum(),
            'pct_outliers': outlier_mask.mean() * 100,
            'lower_bound': lower,
            'upper_bound': upper,
            'outlier_indices': df.index[outlier_mask].tolist()[:10]  # первые 10
        }

    return pd.DataFrame(results).T

# outliers = detect_outliers(df, method='iqr')
# print(outliers[outliers.n_outliers > 0].to_string())
```

---

## 7. Автоматический EDA с ProfileReport

```python
# pip install ydata-profiling
from ydata_profiling import ProfileReport

def generate_profile_report(df: pd.DataFrame, output_file: str = 'eda_report.html'):
    """Полный автоматический EDA отчёт"""
    profile = ProfileReport(
        df,
        title="EDA Report",
        explorative=True,
        minimal=False,  # True для больших датасетов
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
        }
    )
    profile.to_file(output_file)
    print(f"Report saved to {output_file}")
    return profile

# Для Jupyter:
# profile = generate_profile_report(df)
# profile.to_notebook_iframe()
```

---

## 8. Временной анализ данных

```python
def analyze_time_column(df: pd.DataFrame, date_col: str, target: str = None):
    """Анализ временных паттернов"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    print(f"Date range: {df[date_col].min()} → {df[date_col].max()}")
    print(f"Duration: {(df[date_col].max() - df[date_col].min()).days} days")

    # Извлекаем временные признаки
    df['_year'] = df[date_col].dt.year
    df['_month'] = df[date_col].dt.month
    df['_day'] = df[date_col].dt.day
    df['_dayofweek'] = df[date_col].dt.dayofweek
    df['_hour'] = df[date_col].dt.hour if df[date_col].dt.hour.std() > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Количество записей по месяцам
    df.groupby('_month').size().plot(kind='bar', ax=axes[0, 0], title='Records by Month')

    # Количество записей по дням недели
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df.groupby('_dayofweek').size().rename(dict(enumerate(days))).plot(
        kind='bar', ax=axes[0, 1], title='Records by Day of Week'
    )

    if target:
        # Средний таргет по месяцам
        df.groupby('_month')[target].mean().plot(ax=axes[1, 0], title=f'Avg {target} by Month', marker='o')

        # Средний таргет по дням недели
        df.groupby('_dayofweek')[target].mean().rename(dict(enumerate(days))).plot(
            ax=axes[1, 1], title=f'Avg {target} by Day of Week', marker='o'
        )

    plt.tight_layout()
    plt.show()
```
