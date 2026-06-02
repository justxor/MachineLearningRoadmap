# Урок 09: XGBoost, LightGBM, CatBoost

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 08](lesson-08-gradient-boosting-theory.md) | [Урок 10 →](lesson-10-stacking.md)

---

## 🎯 Цели урока

- Понять ключевые улучшения XGBoost над vanilla GBM
- Разобраться в LightGBM: Leaf-wise рост и GOSS
- Освоить CatBoost: Ordered Boosting и обработка категорий
- Научиться подбирать гиперпараметры через Optuna

---

## 📐 Теория: сравнение библиотек

### XGBoost (2016)

**Улучшения над GBM:**
- **Регуляризация:** L1 (α) и L2 (λ) прямо в функции потерь
- **Level-wise рост:** все узлы на глубине d расширяются одновременно
- **Обработка пропусков:** автоматически выбирает оптимальное направление
- **Приближённый алгоритм разбиения:** Weighted Quantile Sketch
- **Параллелизм:** на уровне признаков (не деревьев)

**Функция потерь XGBoost:**
```
Obj = Σ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)

Ω(f) = γT + (λ/2)·Σⱼ wⱼ²    (T = число листьев, w = веса листьев)
```

### LightGBM (2017, Microsoft)

**Ключевые инновации:**
- **Leaf-wise growth:** расширяет лист с максимальным reduction → глубже, точнее, но риск overfit
- **GOSS (Gradient-based One-Side Sampling):** сохраняет все примеры с большим градиентом, сэмплирует остальные
- **EFB (Exclusive Feature Bundling):** объединяет sparse признаки → меньше памяти
- **В 10-20x быстрее XGBoost** при той же точности

### CatBoost (2017, Яндекс)

**Проблема:** Target Leakage при Gradient Boosting с категориальными признаками.

**Решения:**
- **Ordered Boosting:** каждый пример обучается только на предыдущих → нет leakage
- **Ordered Target Encoding:** статистики считаются на случайных перестановках
- **Symmetric Trees (Oblivious):** все листья на одном уровне используют одно условие

---

## 💻 Практика

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Данные с категориальными признаками
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])

# Добавим категориальный признак
X_df['cat_feat'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(X_df))

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# === XGBoost ===
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,     # L1
    reg_lambda=1.0,    # L2
    early_stopping_rounds=50,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train.drop('cat_feat', axis=1), y_train,
              eval_set=[(X_test.drop('cat_feat', axis=1), y_test)],
              verbose=False)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test.drop('cat_feat', axis=1))[:, 1])
print(f"XGBoost AUC: {xgb_auc:.4f}")

# === LightGBM ===
lgb_train = lgb.Dataset(X_train.drop('cat_feat', axis=1), label=y_train)
lgb_valid = lgb.Dataset(X_test.drop('cat_feat', axis=1), label=y_test, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,       # контролирует сложность (leaf-wise)
    'max_depth': -1,        # без ограничения
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,  # аналог colsample_bytree
    'bagging_fraction': 0.8,  # аналог subsample (GOSS)
    'n_jobs': -1,
    'verbose': -1
}
lgb_model = lgb.train(lgb_params, lgb_train,
                       num_boost_round=500,
                       valid_sets=[lgb_valid],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
lgb_auc = roc_auc_score(y_test, lgb_model.predict(X_test.drop('cat_feat', axis=1)))
print(f"LightGBM AUC: {lgb_auc:.4f}")

# === CatBoost (с категориальными признаками!) ===
cat_features = ['cat_feat']
cb_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric='AUC',
    early_stopping_rounds=50,
    random_seed=42,
    verbose=False
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
cb_auc = roc_auc_score(y_test, cb_model.predict_proba(X_test)[:, 1])
print(f"CatBoost AUC: {cb_auc:.4f}")

print(f"\nBest: {'XGBoost' if xgb_auc >= lgb_auc and xgb_auc >= cb_auc else 'LightGBM' if lgb_auc >= cb_auc else 'CatBoost'}")


# === Optuna для подбора гиперпараметров XGBoost ===
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    X_num = X_train.drop('cat_feat', axis=1)
    model.fit(X_num, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test.drop('cat_feat', axis=1))[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(f"\nBest XGBoost AUC (Optuna): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 🔑 Ключевые выводы

| | XGBoost | LightGBM | CatBoost |
|--|---------|----------|---------|
| Рост деревьев | Level-wise | Leaf-wise | Symmetric |
| Скорость | Быстрый | Самый быстрый | Медленнее |
| Категориальные | Вручную | Частично | Автоматически |
| Overfit риск | Средний | Высокий | Низкий |
| Лучше при | Тюнинге | Больших данных | Категориях |

---

## ⚡ Практические задания

1. Benchmark: сравните скорость обучения трёх библиотек при N=1M
2. Сравните Leaf-wise vs Level-wise: при каком max_depth они дают одинаковый результат?
3. Оптимизируйте LightGBM через Optuna с 50 trials. Сколько выиграете у дефолтных параметров?
4. Загрузите Kaggle датасет с категориальными признаками и сравните XGBoost (с OHE) vs CatBoost

---

**[← Урок 08: Gradient Boosting](lesson-08-gradient-boosting-theory.md)** | **[Урок 10: Стекинг →](lesson-10-stacking.md)**
