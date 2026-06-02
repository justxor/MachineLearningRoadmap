# 🛠️ Feature Engineering — Создание Признаков

Готовые сниппеты для создания, трансформации и отбора признаков.

---

## 1. Числовые преобразования

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def add_numeric_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Создаёт трансформированные версии числовых признаков"""
    df = df.copy()
    for col in cols:
        series = df[col]
        # Log transform (для правосторонних распределений)
        if series.min() > 0:
            df[f'{col}_log'] = np.log1p(series)
        # Square root
        if series.min() >= 0:
            df[f'{col}_sqrt'] = np.sqrt(series)
        # Box-Cox / Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        df[f'{col}_boxcox'] = pt.fit_transform(series.values.reshape(-1, 1)).ravel()
        # Binning / квантили
        df[f'{col}_bin'] = pd.qcut(series, q=10, labels=False, duplicates='drop')
        # Квадрат (для нелинейных зависимостей)
        df[f'{col}_sq'] = series ** 2
    return df

# df = add_numeric_features(df, ['age', 'salary', 'tenure'])
```

---

## 2. Взаимодействия признаков

```python
from itertools import combinations

def add_interaction_features(df: pd.DataFrame, cols: list,
                              operations: list = ['multiply', 'divide', 'add']) -> pd.DataFrame:
    """Создаёт попарные взаимодействия признаков"""
    df = df.copy()
    for col1, col2 in combinations(cols, 2):
        if 'multiply' in operations:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        if 'divide' in operations:
            # Защита от деления на ноль
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, np.nan))
        if 'add' in operations:
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        if 'subtract' in operations:
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    return df

# df = add_interaction_features(df, ['height', 'weight', 'age'])
```

---

## 3. Временные признаки

```python
def extract_datetime_features(df: pd.DataFrame, date_col: str,
                               drop_original: bool = False) -> pd.DataFrame:
    """Полный набор временных признаков"""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    # Базовые
    df[f'{date_col}_year'] = dt.dt.year
    df[f'{date_col}_month'] = dt.dt.month
    df[f'{date_col}_day'] = dt.dt.day
    df[f'{date_col}_hour'] = dt.dt.hour
    df[f'{date_col}_minute'] = dt.dt.minute
    df[f'{date_col}_dayofweek'] = dt.dt.dayofweek  # 0=Monday
    df[f'{date_col}_dayofyear'] = dt.dt.dayofyear
    df[f'{date_col}_weekofyear'] = dt.dt.isocalendar().week.astype(int)
    df[f'{date_col}_quarter'] = dt.dt.quarter

    # Бинарные флаги
    df[f'{date_col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df[f'{date_col}_is_month_start'] = dt.dt.is_month_start.astype(int)
    df[f'{date_col}_is_month_end'] = dt.dt.is_month_end.astype(int)
    df[f'{date_col}_is_quarter_end'] = dt.dt.is_quarter_end.astype(int)

    # Циклические признаки (важно для ML!)
    df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    df[f'{date_col}_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df[f'{date_col}_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df[f'{date_col}_dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df[f'{date_col}_dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Разница от текущего момента
    now = pd.Timestamp.now()
    df[f'{date_col}_days_since'] = (now - dt).dt.days

    if drop_original:
        df.drop(columns=[date_col], inplace=True)

    return df

# df = extract_datetime_features(df, 'created_at')
```

---

## 4. Target Encoding (без утечки данных)

```python
from sklearn.model_selection import KFold

def target_encode_kfold(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         cat_col: str, target_col: str,
                         n_splits: int = 5, smoothing: float = 10.0) -> tuple:
    """
    Target encoding с K-Fold для предотвращения утечки данных.
    Smoothing: small group → prior (global mean), large group → group mean.
    """
    global_mean = df_train[target_col].mean()
    new_col = f'{cat_col}_te'

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[new_col] = 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df_train):
        fold_train = df_train.iloc[train_idx]
        # Статистики по фолду
        stats = fold_train.groupby(cat_col)[target_col].agg(['mean', 'count'])
        # Smoothing: combines global mean with group mean
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        df_train.loc[df_train.index[val_idx], new_col] = df_train.iloc[val_idx][cat_col].map(smooth).fillna(global_mean)

    # Для теста используем все тренировочные данные
    stats_full = df_train.groupby(cat_col)[target_col].agg(['mean', 'count'])
    smooth_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
    df_test[new_col] = df_test[cat_col].map(smooth_full).fillna(global_mean)

    return df_train, df_test

# df_train, df_test = target_encode_kfold(df_train, df_test, 'city', 'price')
```

---

## 5. Агрегационные признаки

```python
def add_groupby_features(df: pd.DataFrame, group_cols: list,
                          agg_cols: list, agg_funcs: list = None) -> pd.DataFrame:
    """Создаёт групповые агрегации как признаки"""
    if agg_funcs is None:
        agg_funcs = ['mean', 'median', 'std', 'min', 'max', 'count']

    df = df.copy()
    for group_col in group_cols:
        for agg_col in agg_cols:
            group_stats = df.groupby(group_col)[agg_col].agg(agg_funcs)
            for func in agg_funcs:
                new_col = f'{agg_col}_by_{group_col}_{func}'
                df[new_col] = df[group_col].map(group_stats[func])

            # Разница от среднего по группе (полезно для аномалий)
            group_mean = df.groupby(group_col)[agg_col].transform('mean')
            df[f'{agg_col}_diff_from_{group_col}_mean'] = df[agg_col] - group_mean
            df[f'{agg_col}_ratio_to_{group_col}_mean'] = df[agg_col] / (group_mean + 1e-8)

    return df

# df = add_groupby_features(df, group_cols=['city', 'category'],
#                           agg_cols=['price', 'quantity'])
```

---

## 6. Отбор признаков

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap

def select_features(X_train: pd.DataFrame, y_train: pd.Series,
                    task: str = 'classification', n_features: int = 20) -> dict:
    """Несколько методов отбора признаков"""
    results = {}

    # 1. Корреляция с таргетом
    corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    results['correlation'] = corr.head(n_features).index.tolist()

    # 2. Mutual Information
    if task == 'classification':
        mi_func = mutual_info_classif
        f_func = f_classif
    else:
        mi_func = mutual_info_regression
        f_func = f_regression

    mi_scores = mi_func(X_train.fillna(0), y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    results['mutual_info'] = mi_series.head(n_features).index.tolist()

    # 3. F-statistics
    f_scores, _ = f_func(X_train.fillna(0), y_train)
    f_series = pd.Series(f_scores, index=X_train.columns).sort_values(ascending=False)
    results['f_statistic'] = f_series.head(n_features).index.tolist()

    # 4. Feature Importance (Random Forest)
    if task == 'classification':
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X_train.fillna(0), y_train)
    fi_series = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    results['random_forest'] = fi_series.head(n_features).index.tolist()

    # Консенсус — признаки в топ-N хотя бы в 3 методах из 4
    from collections import Counter
    all_selected = []
    for method_features in results.values():
        all_selected.extend(method_features)
    counts = Counter(all_selected)
    results['consensus'] = [f for f, c in counts.most_common() if c >= 3]

    return results

# selected = select_features(X_train, y_train, task='regression', n_features=20)
# print("Consensus features:", selected['consensus'])
```

---

## 7. Polynomial Features (с контролем взрыва размерности)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, f_regression

def smart_polynomial_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, degree: int = 2,
                                top_percentile: int = 50) -> tuple:
    """Полиномиальные признаки с автоматическим отбором лучших"""
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Отбираем лучшие % признаков
    selector = SelectPercentile(f_regression, percentile=top_percentile)
    X_train_selected = selector.fit_transform(X_train_poly, y_train)
    X_test_selected = selector.transform(X_test_poly)

    feature_names = poly.get_feature_names_out(X_train.columns)
    selected_names = [feature_names[i] for i in selector.get_support(indices=True)]

    print(f"Original: {X_train.shape[1]} → Poly: {X_train_poly.shape[1]} → Selected: {X_train_selected.shape[1]}")
    return (pd.DataFrame(X_train_selected, columns=selected_names),
            pd.DataFrame(X_test_selected, columns=selected_names))
```
