# 🏠 Лаба 01: Predict house prices 🟢

## Цель

Построить регрессионную модель для предсказания цены дома по табличным фичам. Поймёшь, как весь пайплайн выглядит: от EDA до метрики на валидации.

## Датасет

- Kaggle: House Prices - Advanced Regression Techniques.
- 79 признаков, ~1500 строк, таргет SalePrice.
- Альтернатива: California Housing из sklearn.datasets.

## Минимальный пайплайн

1. Загрузка данных, train/val/test split (60/20/20).
2. EDA: распределения, пропуски, выбросы, корреляция с таргетом.
3. Сборка sklearn Pipeline: ColumnTransformer (imputer + OHE + scaler) → GradientBoosting.
4. Обучение, логирование RMSE/MAE.
5. Валидация на hold-out, финальный запуск на test.

## Базовый код

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

num = [c for c in X.columns if X[c].dtype != "object"]
cat = [c for c in X.columns if X[c].dtype == "object"]

pre = ColumnTransformer([
  ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
  ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

pipe = Pipeline([("pre", pre), ("gbr", GradientBoostingRegressor(random_state=42))])
pipe.fit(X_train, np.log1p(y_train))
pred = np.expm1(pipe.predict(X_val))
print("RMSE:", mean_squared_error(y_val, pred, squared=False))
```

## Метрики и baseline

- Основная: RMSE на log(price).
- Дополнительные: MAE, MAPE, R².
- Baseline: среднее значение → RMSE ≈ 80k.
- Linear regression на числовых → RMSE ≈ 35–40k.
- GBR/XGBoost → цель <30k.

## Расширения

- Feature engineering: возраст дома, интеракции площадь×качество.
- Target encoding для категорий высокой кардинальности.
- Стекинг: linear + LightGBM + CatBoost → blend.
- SHAP для интерпретации важности фичей.
- Optuna для подбора гиперпараметров.

## Критерии приёмки

- [ ] EDA в отдельном ноутбуке с выводами.
- [ ] Код обучения в виде .py скрипта, не только ноутбук.
- [ ] Random seed зафиксирован везде.
- [ ] Метрики измерены на hold-out и cross-val.
- [ ] RMSE лучше линейного baseline на ≥5%.
- [ ] README с выводами и графиками.

## Анти-паттерны

- ❌ Применение fit_transform на train+test вместе (leakage).
- ❌ Игнорирование выбросов в таргете.
- ❌ Сравнение моделей без фиксированного split.
- ❌ Обучение на сыром таргете без log-преобразования.

---

[← Назад к Practice Labs](./README.md)
