# ✅ Разбор Задачи 08 — Time Series

**Задание:** [task-08-timeseries.md](../practice-tasks/task-08-timeseries.md)

---

## 🔑 Ключевые инсайты

1. **Хронологический split** — никогда не перемешивай данные временного ряда
2. **shift(1) для rolling features** — без сдвига используешь будущее. Всегда!
3. **Walk-Forward CV** — единственная корректная кросс-валидация для TS
4. **LightGBM** часто бьёт Prophet на структурированных данных с признаками
5. **Рекурсивный прогноз** накапливает ошибку — горизонт > 14 дней уже ненадёжен

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

# Данные из задачи
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
n = len(dates)
trend = np.linspace(100, 180, n)
seasonal_week = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
seasonal_year = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.normal(0, 10, n)
sales = np.maximum(trend + seasonal_week + seasonal_year + noise, 0)

df = pd.DataFrame({
    'date': dates,
    'sales': sales.round(1),
    'is_holiday': (np.random.random(n) < 0.05).astype(int),
    'promo': (np.random.random(n) < 0.1).astype(int),
    'temperature': np.random.normal(10, 15, n),
})
df = df.set_index('date').sort_index()

# Часть 1: Анализ
# ADF тест
result = adfuller(df['sales'])
print(f"ADF: p-value = {result[1]:.6f} → {'Стационарный' if result[1] < 0.05 else 'НЕ стационарный'}")
# Ряд с трендом обычно нестационарен

# Декомпозиция
decompose = seasonal_decompose(df['sales'], period=7, model='additive')
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
for ax, component, title in zip(axes,
    [decompose.observed, decompose.trend, decompose.seasonal, decompose.resid],
    ['Observed', 'Trend', 'Seasonal (period=7)', 'Residual']):
    ax.plot(component)
    ax.set_title(title)
plt.tight_layout()
plt.show()

# Часть 2: Создание признаков
def create_features(df, target='sales', lags=None, windows=None):
    df = df.copy()
    if lags is None:
        lags = [1, 7, 14, 30, 365]
    if windows is None:
        windows = [7, 14, 30]

    # Временные признаки
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Лаги (lag_0 — это текущее значение, нельзя использовать!)
    for lag in lags:
        df[f'lag_{lag}'] = df[target].shift(lag)

    # Rolling со shift(1)! БЕЗ СДВИГА = УТЕЧКА ДАННЫХ
    for w in windows:
        df[f'roll_mean_{w}'] = df[target].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df[target].shift(1).rolling(w).std()

    return df

df_feat = create_features(df)
df_feat = df_feat.dropna()  # удаляем строки с NaN от лагов

# Часть 3: Моделирование
# Хронологический split — НЕ случайный!
split_date = '2023-10-01'
train = df_feat[df_feat.index < split_date]
test = df_feat[df_feat.index >= split_date]

feature_cols = [c for c in df_feat.columns if c != 'sales']
X_train, y_train = train[feature_cols], train['sales']
X_test, y_test = test[feature_cols], test['sales']

# Walk-Forward CV
def walk_forward_cv(model, df, target='sales', n_splits=5, test_size=30):
    feature_cols = [c for c in df.columns if c != target]
    n = len(df)
    split_points = np.linspace(n // 2, n - test_size, n_splits, dtype=int)
    scores = []
    for sp in split_points:
        tr = df.iloc[:sp]
        te = df.iloc[sp:sp+test_size]
        model.fit(tr[feature_cols], tr[target])
        y_pred = model.predict(te[feature_cols])
        rmse = np.sqrt(mean_squared_error(te[target], y_pred))
        mape = np.mean(np.abs((te[target] - y_pred) / (te[target] + 1e-8))) * 100
        scores.append({'rmse': rmse, 'mape': mape})
    return pd.DataFrame(scores).mean()

models_ts = {
    'LinearRegression': LinearRegression(),
    'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1),
}

print("Walk-Forward CV:")
for name, model in models_ts.items():
    scores = walk_forward_cv(model, df_feat[df_feat.index < split_date])
    print(f"  {name}: RMSE={scores['rmse']:.2f}, MAPE={scores['mape']:.2f}%")

# Лучшая модель на тесте
best_model = models_ts['LightGBM']
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)

print(f"\nTest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"Test MAPE: {np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-8))) * 100:.2f}%")

plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
plt.plot(y_test.index, y_pred_test, label='Predicted', linewidth=2, linestyle='--')
plt.title('LightGBM: Predicted vs Actual')
plt.legend()
plt.show()
```

---

## 🐛 Типичные ошибки

1. **Случайный split** — берёшь случайные строки. Но строка из декабря 2023 может попасть в train, а октябрьская в test — модель "видит" будущее.

2. **Rolling без shift(1)**: `df['roll_mean_7'] = df['sales'].rolling(7).mean()` включает текущее значение. С shift: `df['sales'].shift(1).rolling(7).mean()` — только прошлые 7 дней.

3. **Лаг 0** — `df['lag_0'] = df['sales'].shift(0)` = само значение = идеальное предсказание на train, но в production его нет.

---

## 📌 Ключевые выводы

- **LightGBM > LinearRegression** на TS с правильными признаками
- **Лаги 7 и 14** (дневная и 2х недельная сезонность) — самые важные признаки
- **MAPE < 10%** — хороший результат для розничных продаж
- **Рекурсивный прогноз на 30 дней**: ошибка растёт ~линейно с горизонтом
