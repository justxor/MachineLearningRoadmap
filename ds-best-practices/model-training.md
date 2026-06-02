# 🤖 Model Training — Обучение Моделей

Готовые шаблоны для обучения, валидации и тюнинга ML моделей.

---

## 1. Базовый шаблон: Train/Val/Test Split

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df: pd.DataFrame, target_col: str,
                 test_size: float = 0.2, val_size: float = 0.1,
                 stratify: bool = True, random_state: int = 42):
    """Правильное разбиение данных на train/val/test"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_col = y if stratify else None

    # Сначала отделяем test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify_col
    )

    # Из оставшихся отделяем val
    val_relative_size = val_size / (1 - test_size)
    stratify_col2 = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state,
        stratify=stratify_col2
    )

    print(f"Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

    if stratify:
        print(f"\nTarget distribution:")
        print(f"  Train: {y_train.value_counts(normalize=True).to_dict()}")
        print(f"  Val:   {y_val.value_counts(normalize=True).to_dict()}")
        print(f"  Test:  {y_test.value_counts(normalize=True).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test
```

---

## 2. Кросс-валидация — правильный подход

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    StratifiedGroupKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)

def cross_validate_model(model, X, y, task='classification',
                          n_splits=5, groups=None, time_series=False):
    """
    Гибкая кросс-валидация для разных задач.
    groups — для GroupKFold (предотвращает утечку по группам).
    time_series — для временных рядов (сохраняет порядок).
    """
    if time_series:
        cv = TimeSeriesSplit(n_splits=n_splits)
    elif groups is not None:
        if task == 'classification':
            cv = StratifiedGroupKFold(n_splits=n_splits)
        else:
            cv = GroupKFold(n_splits=n_splits)
    else:
        if task == 'classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if task == 'classification':
        scoring = ['accuracy', 'roc_auc', 'f1_weighted']
    else:
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    cv_groups = groups if groups is not None else None
    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        groups=cv_groups,
        n_jobs=-1,
        return_train_score=True,
        verbose=0
    )

    print("Cross-Validation Results:")
    for metric in scoring:
        key = f'test_{metric}'
        values = results[key]
        if 'neg_' in metric:
            values = -values
            metric_name = metric.replace('neg_', '').replace('_', ' ').upper()
        else:
            metric_name = metric.replace('_', ' ').upper()
        print(f"  {metric_name}: {values.mean():.4f} ± {values.std():.4f}")

    return results
```

---

## 3. Optuna — гиперпараметрический тюнинг

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_lightgbm(X_train, y_train, task='classification', n_trials=100, cv=5):
    """Оптимизация LightGBM с Optuna"""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold

    def objective(trial):
        params = {
            'objective': 'binary' if task == 'classification' else 'regression',
            'metric': 'auc' if task == 'classification' else 'rmse',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
        }

        if task == 'classification':
            model = lgb.LGBMClassifier(**params)
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
        else:
            model = lgb.LGBMRegressor(**params)
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_rmse', n_jobs=-1)

        return scores.mean()

    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params
```

---

## 4. OOF Predictions (Out-of-Fold)

```python
def oof_predictions(model_class, model_params: dict, X: np.ndarray, y: np.ndarray,
                     X_test: np.ndarray, n_splits: int = 5,
                     task: str = 'classification') -> tuple:
    """
    Обучение с OOF предсказаниями для стекинга.
    Возвращает (oof_preds, test_preds) без утечки данных.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X)) if task == 'regression' else np.zeros((len(X), 2))
    test_preds = np.zeros((len(X_test), n_splits)) if task == 'regression' else np.zeros((len(X_test), n_splits, 2))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}...", end=' ')
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)

        if task == 'classification':
            oof_preds[val_idx] = model.predict_proba(X_val)
            test_preds[:, fold] = model.predict_proba(X_test)
            score = roc_auc_score(y_val, oof_preds[val_idx, 1])
        else:
            oof_preds[val_idx] = model.predict(X_val)
            test_preds[:, fold] = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))

        scores.append(score)
        print(f"Score: {score:.4f}")

    print(f"\nMean CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    avg_test_preds = test_preds.mean(axis=1) if task == 'regression' else test_preds.mean(axis=1)

    return oof_preds, avg_test_preds
```

---

## 5. Early Stopping с мониторингом

```python
def train_lgbm_with_early_stopping(X_train, y_train, X_val, y_val,
                                    params: dict, task='classification'):
    """LightGBM с early stopping и логированием"""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    default_params = {
        'objective': 'binary' if task == 'classification' else 'regression',
        'metric': 'auc' if task == 'classification' else 'rmse',
        'verbosity': -1,
        'random_state': 42,
    }
    default_params.update(params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")
    return model
```

---

## 6. Шаблон ML эксперимента с MLflow

```python
import mlflow
import mlflow.sklearn
from datetime import datetime

def run_ml_experiment(model, X_train, y_train, X_test, y_test,
                       model_name: str, params: dict,
                       task: str = 'classification'):
    """Шаблон эксперимента с MLflow трекингом"""
    mlflow.set_experiment('my-ds-project')

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"):
        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param('model', model_name)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('n_features', X_train.shape[1])

        # Обучаем
        model.fit(X_train, y_train)

        # Оцениваем и логируем метрики
        if task == 'classification':
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred, average='weighted'),
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_pred = model.predict(X_test)
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }

        mlflow.log_metrics(metrics)

        # Логируем модель
        mlflow.sklearn.log_model(model, artifact_path='model')

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, metrics
```
