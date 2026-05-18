# 📊 Лаба 04: Time-series forecasting спроса 🟡

## Цель

Предсказать спрос на следующие 28 дней по истории продаж. Научиться работать с сезонностью, трендом, праздниками и правильно валидировать.

## Датасет

- M5 Forecasting (Walmart) или Rossmann Store Sales.
- Иерархичные ряды: магазин × товар × дата.

## Минимальный пайплайн

1. EDA: тренд, недельная и годовая сезонность, выбросы.
2. Создание фичей: lags (1,7,28), rolling mean/std, day_of_week, is_holiday, price.
3. Time-series split (expanding window).
4. Baseline: seasonal naive (взять значение неделю назад).
5. LightGBM на лагах.
6. Сравнение с Prophet/ETS/SARIMAX.

## Метрики

- WRMSSE (M5), MAPE, sMAPE.
- Обязательно: backtesting на нескольких окнах.
- Coverage прогнозного интервала (для quantile regression).

## Расширения

- Quantile regression для intervals (П50/П90).
- N-BEATS / Temporal Fusion Transformer.
- Hierarchical reconciliation (top-down vs bottom-up).
- Внешние фичи: погода, акции.

## Критерии приёмки

- [ ] Backtest в минимум 3 окнах.
- [ ] Baseline seasonal naive побит на ≥5% по WRMSSE.
- [ ] График прогноза вместе с фактом в отчёте.
- [ ] Разбор ошибок по сегментам (выходные, праздники).

## Анти-паттерны

- ❌ K-fold split на временных рядах (утечка из будущего).
- ❌ Rolling mean без shift (лик текущего значения).
- ❌ Игнорирование праздников.
- ❌ Метрика только на одном периоде.

---

[← Назад к Practice Labs](./README.md)
