# ⚙️ Preprocessing — Предобработка Данных

Правильная предобработка — основа хорошей модели. Все трансформеры обучаются только на train!

---

## 1. Заполнение пропусков

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Стратегии заполнения пропусков:
    - 'auto': медиана для числовых, мода для категориальных
    - 'knn': K ближайших соседей
    - 'iterative': MICE алгоритм (лучший, но медленный)
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'auto':
        # Числовые — медиана (устойчива к выбросам)
        num_imp = SimpleImputer(strategy='median')
        df[num_cols] = num_imp.fit_transform(df[num_cols])
        # Категориальные — мода
        cat_imp = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    elif strategy == 'knn':
        knn_imp = KNNImputer(n_neighbors=5, weights='uniform')
        df[num_cols] = knn_imp.fit_transform(df[num_cols])
        cat_imp = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    elif strategy == 'iterative':
        # MICE — итеративная импутация (лучшее качество)
        iter_imp = IterativeImputer(max_iter=10, random_state=42, verbose=0)
        df[num_cols] = iter_imp.fit_transform(df[num_cols])
        cat_imp = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # Флаги пропусков (информативны для модели!)
    # for col in original missing cols:
    #     df[f'{col}_was_missing'] = original_missing_mask.astype(int)

    return df
```

---

## 2. Масштабирование признаков

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer
)

def choose_scaler(data: pd.Series, task_hint: str = None):
    """Выбор правильного скейлера по характеристикам данных"""
    has_outliers = (data - data.mean()).abs().max() > 3 * data.std()
    is_sparse = (data == 0).mean() > 0.5

    if is_sparse:
        print("Рекомендуем MaxAbsScaler (сохраняет спарсность)")
        return MaxAbsScaler()
    elif has_outliers:
        print("Рекомендуем RobustScaler (устойчив к выбросам)")
        return RobustScaler()
    elif task_hint == 'neural_net':
        print("Рекомендуем MinMaxScaler для нейросетей: [0, 1]")
        return MinMaxScaler()
    else:
        print("Рекомендуем StandardScaler (универсальный)")
        return StandardScaler()

# Важно: fit только на TRAIN!
def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame,
                   X_test: pd.DataFrame, scaler=None):
    if scaler is None:
        scaler = StandardScaler()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # ТОЛЬКО fit на train, потом transform
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test, scaler
```

---

## 3. Кодирование категориальных признаков

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce  # pip install category_encoders

def encode_categoricals(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         cat_cols: list, target: pd.Series = None,
                         method: str = 'onehot') -> tuple:
    """
    Методы:
    - 'onehot': One-Hot Encoding (для номинальных, мало уникальных)
    - 'ordinal': Ordinal Encoding (для порядковых)
    - 'target': Target Encoding (для высококардинальных + таргет)
    - 'binary': Binary Encoding (компромисс)
    - 'count': Count/Frequency Encoding
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    if method == 'onehot':
        encoder = ce.OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_unknown='ignore')
        df_train = encoder.fit_transform(df_train)
        df_test = encoder.transform(df_test)

    elif method == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])
        df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    elif method == 'target':
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=1.0)
        df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols], target)
        df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    elif method == 'binary':
        encoder = ce.BinaryEncoder(cols=cat_cols)
        df_train = encoder.fit_transform(df_train)
        df_test = encoder.transform(df_test)

    elif method == 'count':
        for col in cat_cols:
            count_map = df_train[col].value_counts()
            df_train[col] = df_train[col].map(count_map).fillna(0)
            df_test[col] = df_test[col].map(count_map).fillna(0)

    return df_train, df_test
```

---

## 4. Обработка выбросов

```python
def handle_outliers(df: pd.DataFrame, cols: list,
                    method: str = 'clip', threshold: float = 3.0) -> pd.DataFrame:
    """
    Методы обработки выбросов:
    - 'clip': обрезать до границ (безопасно)
    - 'remove': удалить строки с выбросами
    - 'winsorize': заменить на перцентили
    """
    df = df.copy()
    for col in cols:
        series = df[col].dropna()

        if method == 'clip':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            df[col] = df[col].clip(lower, upper)

        elif method == 'remove':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            mask = (df[col] >= lower) & (df[col] <= upper)
            df = df[mask]

        elif method == 'winsorize':
            from scipy.stats.mstats import winsorize
            pct = 0.05  # 5% с каждой стороны
            df[col] = winsorize(df[col], limits=[pct, pct])

        elif method == 'zscore':
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col].where(z < threshold, other=df[col].median())

    return df
```

---

## 5. Работа с несбалансированными классами

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter

def handle_imbalanced(X_train: np.ndarray, y_train: np.ndarray,
                       strategy: str = 'smote') -> tuple:
    """
    Стратегии балансировки:
    - 'smote': синтетические примеры (РЕКОМЕНДУЕТСЯ)
    - 'adasyn': адаптивный синтез
    - 'oversample': случайная передискретизация
    - 'undersample': случайная недодискретизация
    - 'smotetomek': SMOTE + Tomek Links (комбо)
    - 'class_weight': веса классов (без изменения данных)
    """
    print(f"До балансировки: {Counter(y_train)}")

    if strategy == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=5)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif strategy == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif strategy == 'smotetomek':
        sampler = SMOTETomek(random_state=42)
    elif strategy == 'class_weight':
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(weights))
        print(f"Веса классов: {class_weights}")
        return X_train, y_train  # Данные не меняем

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"После балансировки: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# ВАЖНО: применяй балансировку ТОЛЬКО к тренировочным данным!
```

---

## 6. Работа с текстовыми данными (базовая очистка)

```python
import re
import unicodedata

def clean_text(text: str, language: str = 'ru') -> str:
    """Базовая очистка текста"""
    if not isinstance(text, str):
        return ''

    # Нормализация Unicode
    text = unicodedata.normalize('NFKC', text)
    # Нижний регистр
    text = text.lower()
    # Удаление URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Удаление email
    text = re.sub(r'\S+@\S+', ' ', text)
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', ' ', text)
    # Удаление спецсимволов (оставляем буквы и пробелы)
    if language == 'ru':
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    else:
        text = re.sub(r'[^a-z\s]', ' ', text)
    # Множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f'{text_col}_clean'] = df[text_col].fillna('').apply(clean_text)
    df[f'{text_col}_len'] = df[f'{text_col}_clean'].str.len()
    df[f'{text_col}_word_count'] = df[f'{text_col}_clean'].str.split().str.len()
    return df
```

---

## 7. Уменьшение памяти DataFrame

```python
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Уменьшает потребление памяти за счёт downcast типов"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        dtype = df[col].dtype

        if dtype == object:
            # Преобразуем в category если уникальных < 50% строк
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

        elif dtype in [np.int8, np.int16, np.int32, np.int64]:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif dtype in [np.float16, np.float32, np.float64]:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32)  # float16 часто нестабилен
            else:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduced)")

    return df
```
