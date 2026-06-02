# ⏱️ Time Series — Временные Ряды

Сниппеты для анализа и прогнозирования временных рядов.

---

## 1. Базовый анализ временного ряда

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_time_series(series: pd.Series, freq: int = None):
    """Полный анализ временного ряда"""
    print(f"Length: {len(series)}")
    print(f"Start: {series.index.min()}")
    print(f"End:   {series.index.max()}")
    print(f"Missing values: {series.isnull().sum()}")
    print(f"Mean: {series.mean():.4f}")
    print(f"Std:  {series.std():.4f}")

    # Тест Дики-Фуллера на стационарность
    result = adfuller(series.dropna())
    print(f"\nADF Test (стационарность):")
    print(f"  Статистика: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  {'СТАЦИОНАРНЫЙ' if result[1] < 0.05 else 'НЕ СТАЦИОНАРНЫЙ'} (α=0.05)")

    # Декомпозиция
    if freq:
        decompose_result = seasonal_decompose(series.dropna(), model='additive', period=freq)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        decompose_result.observed.plot(ax=axes[0], title='Observed')
        decompose_result.trend.plot(ax=axes[1], title='Trend')
        decompose_result.seasonal.plot(ax=axes[2], title='Seasonal')
        decompose_result.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        plt.show()

    # ACF и PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=min(40, len(series)//3), ax=axes[0], title='ACF')
    plot_pacf(series.dropna(), lags=min(40, len(series)//3), ax=axes[1], title='PACF')
    plt.tight_layout()
    plt.show()

    return decompose_result if freq else None
```

---

## 2. Создание признаков для временного ряда

```python
def create_ts_features(df: pd.DataFrame, target_col: str, date_col: str,
                         lags: list = None, windows: list = None) -> pd.DataFrame:
    """Лаговые и оконные признаки для временного ряда"""
    df = df.sort_values(date_col).copy()
    dt = pd.to_datetime(df[date_col])

    # Временные признаки
    df['ts_year'] = dt.dt.year
    df['ts_month'] = dt.dt.month
    df['ts_day'] = dt.dt.day
    df['ts_dayofweek'] = dt.dt.dayofweek
    df['ts_hour'] = dt.dt.hour
    df['ts_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['ts_quarter'] = dt.dt.quarter
    df['ts_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['ts_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    df['ts_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['ts_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

    # Лаговые признаки
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Скользящие статистики
    if windows is None:
        windows = [3, 7, 14, 28]
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        df[f'{target_col}_roll_min_{window}'] = df[target_col].shift(1).rolling(window).min()
        df[f'{target_col}_roll_max_{window}'] = df[target_col].shift(1).rolling(window).max()
        df[f'{target_col}_roll_median_{window}'] = df[target_col].shift(1).rolling(window).median()

    # Экспоненциальное сглаживание
    for alpha in [0.3, 0.5]:
        df[f'{target_col}_ewm_{alpha}'] = df[target_col].shift(1).ewm(alpha=alpha).mean()

    # Разности (дифференцирование)
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_7'] = df[target_col].diff(7)

    return df
```

---

## 3. Walk-Forward Validation (правильная CV для TS)

```python
def walk_forward_validation(df: pd.DataFrame, target_col: str,
                              model, n_splits: int = 5,
                              test_size: int = 30) -> dict:
    """
    Правильная кросс-валидация для временных рядов.
    Нельзя использовать обычный KFold (утечка из будущего)!
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    errors_mae = []
    errors_rmse = []
    feature_cols = [c for c in df.columns if c != target_col]
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    split_points = np.linspace(n // 2, n - test_size, n_splits, dtype=int)

    for i, split_point in enumerate(split_points):
        train = df.iloc[:split_point]
        test = df.iloc[split_point:split_point + test_size]

        if len(test) == 0:
            continue

        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        errors_mae.append(mae)
        errors_rmse.append(rmse)

        print(f"Split {i+1}: train={len(train)}, test={len(test)}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    print(f"\nMean MAE: {np.mean(errors_mae):.4f} ± {np.std(errors_mae):.4f}")
    print(f"Mean RMSE: {np.mean(errors_rmse):.4f} ± {np.std(errors_rmse):.4f}")

    return {'mae': errors_mae, 'rmse': errors_rmse}
```

---

## 4. Forecast с Prophet

```python
# pip install prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

def forecast_with_prophet(df: pd.DataFrame, date_col: str, value_col: str,
                            periods: int = 30, freq: str = 'D',
                            seasonalities: dict = None) -> pd.DataFrame:
    """Прогноз временного ряда с Facebook Prophet"""
    prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})

    model = Prophet(
        seasonality_mode='multiplicative',  # или 'additive'
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # чем больше — тем гибче тренд
        seasonality_prior_scale=10.0,
    )

    if seasonalities:
        for name, params in seasonalities.items():
            model.add_seasonality(name=name, **params)

    model.fit(prophet_df)

    # Создаём будущие даты
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Визуализация
    fig1 = model.plot(forecast, figsize=(14, 6))
    plt.title('Prophet Forecast')
    plt.tight_layout()
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.show()

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
```

---

## 5. Обнаружение аномалий в временном ряду

```python
def detect_ts_anomalies(series: pd.Series, window: int = 30,
                          threshold: float = 3.0) -> pd.Series:
    """Обнаружение аномалий через скользящее Z-score"""
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

    z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
    anomalies = z_scores.abs() > threshold

    # Визуализация
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(series, label='Original', alpha=0.7)
    axes[0].plot(rolling_mean, label='Rolling Mean', color='orange')
    axes[0].fill_between(series.index,
                         rolling_mean - threshold * rolling_std,
                         rolling_mean + threshold * rolling_std,
                         alpha=0.2, color='orange', label='Bounds')
    axes[0].scatter(series[anomalies].index, series[anomalies],
                   color='red', s=50, zorder=5, label='Anomaly')
    axes[0].legend()
    axes[0].set_title('Time Series with Anomalies')

    axes[1].plot(z_scores, label='Z-score', color='purple')
    axes[1].axhline(y=threshold, color='red', linestyle='--')
    axes[1].axhline(y=-threshold, color='red', linestyle='--')
    axes[1].set_title('Z-score')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Anomalies found: {anomalies.sum()} ({anomalies.mean()*100:.2f}%)")
    return anomalies
```
