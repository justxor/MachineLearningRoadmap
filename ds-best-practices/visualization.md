# 📈 Visualization — Визуализация Данных

Готовые шаблоны для создания профессиональных графиков в DS проектах.

---

## 0. Настройка стилей

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matplotlib настройки
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 120,
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Цветовые палитры для DS
COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#3498db',
    'highlight': '#f39c12',
    'secondary': '#9b59b6',
}
```

---

## 1. Scatter Plot с регрессионной линией

```python
def plot_scatter_with_regression(df, x_col, y_col, hue_col=None,
                                  title=None, alpha=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Matplotlib версия
    ax = axes[0]
    if hue_col:
        for category in df[hue_col].unique():
            mask = df[hue_col] == category
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                      alpha=alpha, label=category, s=30)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=alpha, color='steelblue', s=30)

    # Линия регрессии
    import numpy as np
    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
    p = np.poly1d(z)
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', linewidth=2, label='Trend')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'{y_col} vs {x_col}')
    if hue_col:
        ax.legend()

    # Seaborn версия
    sns.regplot(data=df, x=x_col, y=y_col, ax=axes[1],
                scatter_kws={'alpha': alpha, 's': 30},
                line_kws={'color': 'red', 'linestyle': '--'})
    axes[1].set_title(f'Seaborn: {y_col} vs {x_col}')

    plt.tight_layout()
    plt.show()

    # Корреляция
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

# plot_scatter_with_regression(df, 'age', 'salary', hue_col='department')
```

---

## 2. Plotly — Интерактивные графики

```python
def interactive_scatter(df, x_col, y_col, color_col=None, size_col=None,
                         hover_cols=None, title='Scatter Plot'):
    """Интерактивный scatter plot с Plotly"""
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        size=size_col,
        hover_data=hover_cols,
        title=title,
        opacity=0.7,
        template='plotly_white',
        trendline='ols',  # линия тренда
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        title_font_size=16,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    fig.show()

def interactive_histogram(df, col, nbins=50, color_col=None, title=None):
    """Интерактивная гистограмма"""
    fig = px.histogram(
        df, x=col, nbins=nbins,
        color=color_col,
        marginal='box',  # boxplot сверху
        title=title or f'Distribution of {col}',
        template='plotly_white',
        opacity=0.8,
    )
    fig.show()

def interactive_heatmap(corr_matrix, title='Correlation Matrix'):
    """Интерактивная тепловая карта корреляций"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        showscale=True,
    ))
    fig.update_layout(title=title, template='plotly_white')
    fig.show()
```

---

## 3. Анализ моделей — Feature Importance

```python
def plot_feature_importance(model, feature_names, top_n=20, title='Feature Importance'):
    """Визуализация важности признаков"""
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(
        range(top_n),
        importances[indices[::-1]],
        color=plt.cm.RdYlGn(importances[indices[::-1]] / importances.max())
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance')
    ax.set_title(title)

    # Добавляем значения на бары
    for bar, imp in zip(bars, importances[indices[::-1]]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_shap_summary(model, X, max_display=20):
    """SHAP summary plot"""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, max_display=max_display, show=True)
```

---

## 4. ROC и Precision-Recall кривые

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_pr_curves(y_true, y_scores_dict, figsize=(14, 6)):
    """
    y_scores_dict: {'Model Name': y_proba, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    ax = axes[0]
    for name, scores in y_scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

    # Precision-Recall Curve
    ax = axes[1]
    for name, scores in y_scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap:.3f})')

    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Использование:
# plot_roc_pr_curves(y_test, {
#     'LightGBM': lgb_proba,
#     'XGBoost': xgb_proba,
#     'Logistic': lr_proba
# })
```

---

## 5. Матрица ошибок (Confusion Matrix)

```python
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, figsize=(8, 6)):
    """Красивая матрица ошибок"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    if labels is not None:
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=12)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    title = 'Confusion Matrix' + (' (Normalized)' if normalize else '')
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
```

---

## 6. Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1,
                         train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'):
    """Кривые обучения для диагностики bias/variance"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        return_times=False
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Train score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Val score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='orange')

    gap = train_mean[-1] - val_mean[-1]
    ax.set_title(f'Learning Curves | Final gap: {gap:.3f}\n'
                 f'({("High Variance/Overfit" if gap > 0.05 else "Good fit") if val_mean[-1] > 0.7 else "High Bias/Underfit"})',
                 fontsize=12)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(scoring.upper())
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()
```

---

## 7. Визуализация кластеров (2D/3D)

```python
def plot_clusters(X_2d: np.ndarray, labels: np.ndarray,
                   method_name: str = 'Clusters', figsize=(10, 8)):
    """Визуализация результатов кластеризации"""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    palette = sns.color_palette('husl', n_clusters)

    fig, ax = plt.subplots(figsize=figsize)
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        color = 'gray' if cluster_id == -1 else palette[cluster_id]
        label = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[color], s=30, alpha=0.7, label=label)

    ax.set_title(f'{method_name} | {n_clusters} clusters', fontsize=14)
    ax.legend(markerscale=1.5, framealpha=0.9)
    plt.tight_layout()
    plt.show()

# from sklearn.manifold import TSNE
# X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
# plot_clusters(X_2d, kmeans.labels_, 'K-Means (TSNE)')
```
