# 🔗 Pipelines — ML Пайплайны и Автоматизация

sklearn Pipeline — стандарт индустрии для создания воспроизводимых ML пайплайнов.

---

## 1. Базовый Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def build_sklearn_pipeline(model=None, numeric_strategy='median',
                             categorical_strategy='onehot',
                             scale=True):
    """Универсальный sklearn Pipeline"""
    # Числовые признаки
    num_steps = [('imputer', SimpleImputer(strategy=numeric_strategy))]
    if scale:
        num_steps.append(('scaler', RobustScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Категориальные признаки
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Колонки по типу
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
    ], remainder='drop', n_jobs=-1)

    if model is None:
        model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

# Использование:
# pipe = build_sklearn_pipeline()
# pipe.fit(X_train, y_train)
# pipe.predict(X_test)
# cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

---

## 2. Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Кастомный трансформер: извлечение временных признаков"""

    def __init__(self, date_col: str, features: list = None):
        self.date_col = date_col
        self.features = features or ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']

    def fit(self, X, y=None):
        return self  # Не нужен fit для stateless трансформеров

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.date_col])

        feature_map = {
            'year': dt.dt.year,
            'month': dt.dt.month,
            'day': dt.dt.day,
            'dayofweek': dt.dt.dayofweek,
            'hour': dt.dt.hour,
            'minute': dt.dt.minute,
            'is_weekend': (dt.dt.dayofweek >= 5).astype(int),
            'month_sin': np.sin(2 * np.pi * dt.dt.month / 12),
            'month_cos': np.cos(2 * np.pi * dt.dt.month / 12),
        }

        for feat in self.features:
            if feat in feature_map:
                X[f'{self.date_col}_{feat}'] = feature_map[feat]

        X = X.drop(columns=[self.date_col])
        return X

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """Кастомный трансформер: Target Encoding (без утечки через Pipeline)"""

    def __init__(self, cols: list, smoothing: float = 10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.encoding_maps = {}
        self.global_mean = None

    def fit(self, X, y):
        df = pd.DataFrame(X)
        self.global_mean = y.mean()
        for col in self.cols:
            if col in df.columns:
                stats = df.groupby(col)[y].agg(['mean', 'count'])
                smooth = ((stats['count'] * stats['mean'] + self.smoothing * self.global_mean) /
                          (stats['count'] + self.smoothing))
                self.encoding_maps[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, mapping in self.encoding_maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(self.global_mean)
        return X.values
```

---

## 3. Pipeline с GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_pipeline(X_train, y_train, task='classification'):
    """Pipeline + RandomizedSearch для автоматического тюнинга"""
    pipe = build_sklearn_pipeline()

    # Параметры для поиска — используем 'model__param' нотацию
    param_distributions = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [3, 5, 7, None],
        'model__min_samples_leaf': [1, 5, 10],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 1.0],
    }

    scoring = 'roc_auc' if task == 'classification' else 'neg_rmse'

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True,  # Переобучаем на лучших параметрах на всех данных
    )

    search.fit(X_train, y_train)
    print(f"Best score: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_
```

---

## 4. Сохранение и загрузка Pipeline

```python
import joblib
import pickle
from pathlib import Path

def save_pipeline(pipeline, filepath: str, metadata: dict = None):
    """Сохранение пайплайна с метаданными"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_obj = {
        'pipeline': pipeline,
        'metadata': metadata or {},
        'sklearn_version': __import__('sklearn').__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'saved_at': pd.Timestamp.now().isoformat(),
    }

    joblib.dump(save_obj, filepath, compress=('zlib', 3))
    size = filepath.stat().st_size / 1024
    print(f"Saved to {filepath} ({size:.1f} KB)")

def load_pipeline(filepath: str):
    """Загрузка пайплайна с проверкой версий"""
    save_obj = joblib.load(filepath)
    print(f"Model saved at: {save_obj.get('metadata', {}).get('saved_at', 'unknown')}")
    return save_obj['pipeline']

# save_pipeline(best_pipe, 'models/best_model_v1.pkl', metadata={'version': '1.0', 'author': 'your_name'})
# pipe = load_pipeline('models/best_model_v1.pkl')
```

---

## 5. Feature Union — несколько источников признаков

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Выбор колонок по имени для использования в Pipeline"""
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns] if isinstance(X, pd.DataFrame) else X[:, self.columns]

# Пример: объединяем числовые признаки + PCA + текстовые TF-IDF
def build_multimodal_pipeline():
    numeric_pipeline = Pipeline([
        ('selector', ColumnSelector(['age', 'salary', 'experience'])),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    text_pipeline = Pipeline([
        ('selector', ColumnSelector(['description'])),
        ('flatten', FunctionTransformer(lambda x: x.values.ravel())),
        ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
    ])

    feature_union = FeatureUnion([
        ('numeric', numeric_pipeline),
        ('text', text_pipeline),
    ])

    full_pipeline = Pipeline([
        ('features', feature_union),
        ('pca', PCA(n_components=50)),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    return full_pipeline
```
