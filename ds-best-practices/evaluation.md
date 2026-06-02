# 📏 Evaluation — Оценка Качества Моделей

Полный арсенал метрик и инструментов для правильной оценки ML моделей.

---

## 1. Полный отчёт — Классификация

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix,
    log_loss, brier_score_loss
)

def classification_full_report(y_true, y_pred, y_proba=None,
                                 class_names=None, verbose=True):
    """Полный отчёт по классификации"""
    results = {}

    # Базовые метрики
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    results['mcc'] = matthews_corrcoef(y_true, y_pred)
    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    if y_proba is not None:
        is_binary = len(np.unique(y_true)) == 2
        if is_binary:
            proba = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
            results['roc_auc'] = roc_auc_score(y_true, proba)
            results['pr_auc'] = average_precision_score(y_true, proba)
            results['log_loss'] = log_loss(y_true, proba)
            results['brier_score'] = brier_score_loss(y_true, proba)
        else:
            results['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            results['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            results['log_loss'] = log_loss(y_true, y_proba)

    if verbose:
        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k:25s}: {v:.4f}")
        print("\nDetailed Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return results
```

---

## 2. Полный отчёт — Регрессия

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)

def regression_full_report(y_true, y_pred, verbose=True):
    """Полный отчёт по регрессии"""
    results = {}

    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['medae'] = median_absolute_error(y_true, y_pred)
    results['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100  # в %
    results['r2'] = r2_score(y_true, y_pred)
    results['explained_variance'] = explained_variance_score(y_true, y_pred)
    results['max_error'] = max_error(y_true, y_pred)

    # Нормализованный RMSE
    results['nrmse'] = results['rmse'] / (y_true.max() - y_true.min())

    # Анализ остатков
    residuals = y_true - y_pred
    results['residuals_mean'] = residuals.mean()
    results['residuals_std'] = residuals.std()
    results['residuals_skew'] = pd.Series(residuals).skew()

    if verbose:
        print("=" * 50)
        print("REGRESSION REPORT")
        print("=" * 50)
        for k, v in results.items():
            unit = '%' if k == 'mape' else ''
            print(f"  {k:25s}: {v:.4f}{unit}")

    return results
```

---

## 3. Bootstrap доверительные интервалы для метрик

```python
def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000,
                      ci=0.95, random_state=42):
    """
    Доверительный интервал для любой метрики через Bootstrap.
    Важно: помогает понять насколько стабильна ваша оценка.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    metric_scores = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        score = metric_func(y_true[indices], y_pred[indices])
        metric_scores.append(score)

    metric_scores = np.array(metric_scores)
    alpha = (1 - ci) / 2

    point_estimate = metric_func(y_true, y_pred)
    lower = np.percentile(metric_scores, alpha * 100)
    upper = np.percentile(metric_scores, (1 - alpha) * 100)

    print(f"Metric: {point_estimate:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    return point_estimate, lower, upper

# Пример:
# from sklearn.metrics import roc_auc_score
# bootstrap_metric(y_test, y_pred_proba, roc_auc_score)
```

---

## 4. Анализ ошибок модели

```python
def error_analysis(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                    features: list, task: str = 'classification') -> pd.DataFrame:
    """Детальный анализ ошибок — где модель ошибается?"""
    df = df.copy()
    df['_is_correct'] = (df[y_true_col] == df[y_pred_col]) if task == 'classification' else True
    df['_error'] = df[y_true_col] - df[y_pred_col]
    df['_abs_error'] = df['_error'].abs()

    if task == 'classification':
        errors = df[~df['_is_correct']].copy()
        print(f"Total errors: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.1f}%)")

        # Анализ ошибок по признакам
        for feat in features[:5]:
            if df[feat].dtype in ['object', 'category']:
                error_by_feat = errors.groupby(feat).size() / df.groupby(feat).size()
                print(f"\nError rate by {feat}:")
                print(error_by_feat.sort_values(ascending=False).head(10))

    else:  # Регрессия
        print(f"Max error: {df['_abs_error'].max():.2f}")
        print(f"Top-10 worst predictions:")
        worst = df.nlargest(10, '_abs_error')[[y_true_col, y_pred_col, '_error', '_abs_error'] + features[:3]]
        print(worst.to_string())

    return df

def find_systematic_errors(y_true, y_pred, X: pd.DataFrame, threshold=0.3):
    """Поиск систематических ошибок в разных сегментах данных"""
    residuals = y_true - y_pred
    results = []

    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            # Числовые — корреляция с ошибками
            corr = np.corrcoef(X[col].fillna(0), residuals)[0, 1]
            if abs(corr) > threshold:
                results.append({'feature': col, 'correlation_with_error': corr, 'type': 'numeric'})
        else:
            # Категориальные — дисперсия ошибок по группам
            for cat in X[col].unique():
                mask = X[col] == cat
                if mask.sum() > 30:
                    group_mean_error = residuals[mask].mean()
                    if abs(group_mean_error) > threshold:
                        results.append({'feature': f'{col}={cat}', 'mean_error': group_mean_error, 'count': mask.sum(), 'type': 'categorical'})

    return pd.DataFrame(results).sort_values('correlation_with_error' if 'correlation_with_error' in pd.DataFrame(results).columns else 'mean_error', key=abs, ascending=False)
```

---

## 5. Сравнение моделей

```python
def compare_models(models: dict, X_train, y_train, X_test, y_test,
                    task='classification', cv=5) -> pd.DataFrame:
    """
    models: {'Model Name': model_instance}
    Обучает все модели, сравнивает на CV и тесте.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    results = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if task == 'classification' else KFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...", end='')

        if task == 'classification':
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
            model.fit(X_train, y_train)
            test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            metric = 'ROC-AUC'
        else:
            cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
            model.fit(X_train, y_train)
            test_score = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            metric = 'RMSE'

        result = {
            'model': name,
            f'cv_{metric}_mean': cv_scores.mean(),
            f'cv_{metric}_std': cv_scores.std(),
            f'test_{metric}': test_score,
            'overfit_gap': abs(cv_scores.mean() - test_score),
        }
        results.append(result)
        print(f" CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | Test: {test_score:.4f}")

    df_results = pd.DataFrame(results).sort_values(f'cv_{metric}_mean',
                                                     ascending=(metric == 'RMSE'))
    return df_results

# Использование:
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from lightgbm import LGBMClassifier
# models = {
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'GBM': GradientBoostingClassifier(random_state=42),
#     'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
# }
# results = compare_models(models, X_train, y_train, X_test, y_test)
```

---

## 6. Калибровка вероятностей

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_and_plot(model, X_train, y_train, X_test, y_test,
                        method='isotonic', cv=5):
    """Калибровка вероятностей и визуализация"""
    calibrated_model = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated_model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(8, 8))

    # До калибровки
    model.fit(X_train, y_train)
    prob_true_orig, prob_pred_orig = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    ax.plot(prob_pred_orig, prob_true_orig, 's-', label='Original', color='red')

    # После калибровки
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_model.predict_proba(X_test)[:, 1], n_bins=10)
    ax.plot(prob_pred_cal, prob_true_cal, 's-', label=f'Calibrated ({method})', color='green')

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return calibrated_model
```
