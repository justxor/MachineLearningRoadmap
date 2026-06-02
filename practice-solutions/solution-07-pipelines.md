# ✅ Разбор Задачи 07 — ML Pipelines

**Задание:** [task-07-pipelines.md](../practice-tasks/task-07-pipelines.md)

---

## 🔑 Ключевые инсайты

1. **Pipeline защищает от leakage** — всё что внутри pipeline автоматически fit только на train
2. **Кастомный трансформер** — наследуй от BaseEstimator + TransformerMixin и реализуй fit/transform
3. **step__param нотация** — для GridSearch через Pipeline
4. **handle_unknown='ignore'** — для production: новые категории → нули в OHE (не падает)
5. **Joblib сжимает** — compress=3 снижает размер на 40–70%

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Данные
np.random.seed(42)
n = 3000
df = pd.DataFrame({
    'area_sqm':       np.abs(np.random.normal(70, 30, n)),
    'rooms':          np.random.randint(1, 6, n),
    'floor':          np.random.randint(1, 25, n),
    'total_floors':   np.random.randint(1, 25, n),
    'build_year':     np.random.randint(1960, 2024, n),
    'district':       np.random.choice(['Center','North','South','East','West'], n),
    'condition':      np.random.choice(['New','Good','Fair','Poor'], n, p=[0.2,0.4,0.3,0.1]),
    'has_parking':    np.random.binomial(1, 0.4, n),
    'distance_metro': np.abs(np.random.exponential(1.5, n)),
    'price_million':  np.abs(np.random.normal(8, 4, n)),
})
df.loc[np.random.choice(n, 100), 'area_sqm'] = np.nan
df.loc[np.random.choice(n, 80), 'build_year'] = np.nan
df.loc[np.random.choice(n, 60), 'district'] = np.nan
df.loc[5, 'floor'] = 99
df.loc[10, 'area_sqm'] = 0

X = df.drop(columns=['price_million'])
y = df['price_million']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Кастомные трансформеры ===

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=3.0):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = {}
        self.upper_ = {}
        for col in X.columns:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.multiplier * iqr
            self.upper_[col] = q3 + self.multiplier * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.lower_:
                X[col] = X[col].clip(self.lower_[col], self.upper_[col])
        return X.values

class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, year_col='build_year', current_year=2024):
        self.year_col = year_col
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.year_col in X.columns:
            X['building_age'] = self.current_year - X[self.year_col]
            X = X.drop(columns=[self.year_col])
        return X

class FloorRatioAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if 'floor' in X.columns and 'total_floors' in X.columns:
            X['floor_ratio'] = X['floor'] / (X['total_floors'] + 1e-6)
            X['is_top_floor'] = (X['floor'] == X['total_floors']).astype(int)
        return X

# === Полный Pipeline ===
num_cols = ['area_sqm', 'rooms', 'floor', 'total_floors', 'has_parking', 'distance_metro']
cat_cols = ['district', 'condition']

numeric_transformer = Pipeline([
    ('clipper', OutlierClipper(multiplier=3.0)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
])

# Добавляем AgeCalculator ДО ColumnTransformer (трансформирует весь датафрейм)
full_pipeline = Pipeline([
    ('floor_ratio', FloorRatioAdder()),
    ('age_calc', AgeCalculator()),
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0)),
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")

# GridSearchCV через pipeline
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'model__alpha': [0.1, 1.0, 10.0, 100.0],
}
gs = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='neg_rmse', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Best CV RMSE: {-gs.best_score_:.4f}")

# Сохранение с метаданными
save_obj = {
    'pipeline': gs.best_estimator_,
    'metadata': {
        'version': '1.0',
        'rmse_val': -gs.best_score_,
        'rmse_test': np.sqrt(mean_squared_error(y_test, gs.predict(X_test))),
        'features': list(X_train.columns),
        'saved_at': pd.Timestamp.now().isoformat(),
    }
}
joblib.dump(save_obj, 'real_estate_pipeline_v1.pkl', compress=3)
print("Saved!")

# Загрузка и проверка
loaded = joblib.load('real_estate_pipeline_v1.pkl')
pipe_loaded = loaded['pipeline']
y_pred_loaded = pipe_loaded.predict(X_test)
assert np.allclose(y_pred_loaded, gs.predict(X_test)), "Predictions differ!"
print("Pipeline loaded and verified!")

# Production: новые данные с новой категорией district='Airport'
new_data = pd.DataFrame({
    'area_sqm': [65.0], 'rooms': [3], 'floor': [5], 'total_floors': [10],
    'build_year': [2015], 'district': ['Airport'],  # новая категория!
    'condition': ['Good'], 'has_parking': [1], 'distance_metro': [2.5]
})
pred = pipe_loaded.predict(new_data)
print(f"Prediction for new data (unknown district): {pred[0]:.2f} млн руб")
# handle_unknown='ignore' → новая категория → все нули → модель всё равно работает
```

---

## 🐛 Типичные ошибки

1. **Не использовать Pipeline** — fit scaler/imputer вручную и забыть сохранить их для inference
2. **Кастомный трансформер без BaseEstimator** — тогда не работает clone() и GridSearch
3. **ColumnTransformer без remainder** — по умолчанию remainder='drop' → новые колонки потеряются
4. **Не проверять predict на новых данных** — модель в production получает другие данные

---

## 📌 Ключевые выводы

- **Pipeline = самодокументирующийся код** — весь preprocessing + model в одном объекте
- **Кастомные трансформеры** — ключ к сложным domain-specific преобразованиям
- **handle_unknown='ignore'** — обязателен для production OHE
- **Версионирование** — всегда сохраняй метрики вместе с моделью
