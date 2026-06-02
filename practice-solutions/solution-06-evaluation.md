# ✅ Разбор Задачи 06 — Evaluation

**Задание:** [task-06-evaluation.md](../practice-tasks/task-06-evaluation.md)

---

## 🔑 Ключевые инсайты

1. **ROC-AUC не зависит от threshold** — хорошо для сравнения моделей
2. **PR-AUC важнее ROC-AUC при дисбалансе классов** — ROC-AUC оптимистичен при 5% позитивных
3. **Cost matrix** — переводи метрики в деньги для бизнес-решений
4. **Bootstrap CI** — оценивай не только точку, но и разброс метрики
5. **MCC** — лучшая одна метрика при дисбалансе (учитывает все 4 квадранта матрицы)

---

## 💻 Полное решение

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Данные из задачи
np.random.seed(42)
n = 2000
y_true = np.random.binomial(1, 0.2, n)
y_proba_A = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, n), 0, 1)
y_proba_B = np.clip(y_true * 0.6 + np.random.normal(0, 0.15, n), 0, 1)
y_proba_C = np.clip(y_true * 0.5 + np.random.normal(0, 0.3, n), 0, 1)
threshold = 0.5
y_pred_A = (y_proba_A >= threshold).astype(int)
y_pred_B = (y_proba_B >= threshold).astype(int)
y_pred_C = (y_proba_C >= threshold).astype(int)

# Часть 1: Полное сравнение метрик
results = {}
for name, y_pred, y_proba in [('A', y_pred_A, y_proba_A), ('B', y_pred_B, y_proba_B), ('C', y_pred_C, y_proba_C)]:
    results[f'Model {name}'] = {
        'Accuracy':          accuracy_score(y_true, y_pred),
        'Balanced Acc':      balanced_accuracy_score(y_true, y_pred),
        'Precision':         precision_score(y_true, y_pred, zero_division=0),
        'Recall':            recall_score(y_true, y_pred, zero_division=0),
        'F1':                f1_score(y_true, y_pred, zero_division=0),
        'MCC':               matthews_corrcoef(y_true, y_pred),
        'ROC-AUC':           roc_auc_score(y_true, y_proba),
        'PR-AUC':            average_precision_score(y_true, y_proba),
        'Log Loss':          log_loss(y_true, y_proba),
        'Brier Score':       brier_score_loss(y_true, y_proba),
    }

df_results = pd.DataFrame(results).T
print(df_results.round(4).to_string())
# Вывод: Model A — лучший Recall. Model B — лучший AUC. Model C — слабая по всем.

# Часть 2: ROC и PR кривые
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, y_proba, color in [('A', y_proba_A, 'blue'), ('B', y_proba_B, 'green'), ('C', y_proba_C, 'red')]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f'Model {name} (AUC={auc:.3f})')

    p, r, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(r, p, color=color, linewidth=2, label=f'Model {name} (AP={ap:.3f})')

axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title('ROC Curves')
axes[0].legend()

baseline = y_true.mean()
axes[1].axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.2f})')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()
plt.tight_layout()
plt.show()

# Часть 3: Threshold optimization
precision_B, recall_B, thresholds_B = precision_recall_curve(y_true, y_proba_B)

# Максимальный F1
f1_scores = 2 * precision_B[:-1] * recall_B[:-1] / (precision_B[:-1] + recall_B[:-1] + 1e-8)
best_t_f1 = thresholds_B[np.argmax(f1_scores)]
print(f"\nBest F1 threshold: {best_t_f1:.3f} → F1={f1_scores.max():.3f}")

# Топ-300 из n=2000 → 15% → найти threshold при котором 300 помечено как 1
n_budget = 300
thresholds_sorted = np.sort(y_proba_B)[::-1]
budget_threshold = thresholds_sorted[min(n_budget, len(thresholds_sorted)) - 1]
print(f"Budget threshold (top 300): {budget_threshold:.3f}")
y_pred_budget = (y_proba_B >= budget_threshold).astype(int)
print(f"  Precision: {precision_score(y_true, y_pred_budget):.3f}")
print(f"  Recall: {recall_score(y_true, y_pred_budget):.3f}")

# Cost matrix: FN=100, FP=10
costs = []
for t in np.arange(0.1, 0.9, 0.01):
    y_pred_t = (y_proba_B >= t).astype(int)
    cm = confusion_matrix(y_true, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * 100 + fp * 10
    costs.append((t, total_cost))
best_cost_t, min_cost = min(costs, key=lambda x: x[1])
print(f"\nCost-optimal threshold: {best_cost_t:.2f} → Cost={min_cost:.0f} руб")

# Часть 4: Bootstrap CI
def bootstrap_auc(y_true, y_proba, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    scores = [roc_auc_score(y_true[idx := rng.choice(len(y_true), len(y_true))], y_proba[idx])
              for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(scores, [alpha*100, (1-alpha)*100])

point, (lo, hi) = bootstrap_auc(y_true, y_proba_B)
print(f"\nModel B ROC-AUC: {point:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")
```

---

## 🐛 Типичные ошибки

1. **Путать Precision и Recall** — Precision = из предсказанных позитивных, сколько правда. Recall = из всех реально позитивных, сколько нашли.
2. **Не строить PR-AUC при дисбалансе** — ROC-AUC 0.85 при 5% позитивных может скрывать плохую PR-AUC.
3. **Не учитывать cost matrix** — F1 не всегда соответствует бизнес-цели.
4. **Пренебрегать Bootstrap CI** — точечная оценка может быть случайной.

---

## 📌 Итоговая рекомендация

**Для задачи оттока клиентов:**
- Выбери **Модель A** (лучший Recall) если важно не пропустить уходящих
- Выбери **Модель B** (лучший AUC) если нужен баланс
- Threshold: ~0.35–0.40 для максимизации Recall
- **Метрика для мониторинга:** PR-AUC (не ROC-AUC, т.к. дисбаланс)
