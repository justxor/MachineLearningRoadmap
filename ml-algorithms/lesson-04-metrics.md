# Урок 04: Метрики качества ML моделей

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 03](lesson-03-logistic-regression.md) | [Урок 05 →](lesson-05-decision-trees.md)

---

## 🎯 Цели урока

- Понять почему Accuracy не всегда подходит
- Разобраться в Precision/Recall/F1 и когда что использовать
- Научиться читать ROC-AUC и PR-кривые
- Освоить метрики регрессии: MAE, MSE, RMSE, R², MAPE

---

## 📐 Метрики классификации

### Confusion Matrix

```
               Предсказанный класс
                  Positive | Negative
Реальный Positive    TP    |    FN
         Negative    FP    |    TN
```

### Основные метрики

| Метрика | Формула | Когда использовать |
|---------|---------|-------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Балансированные классы |
| Precision | TP/(TP+FP) | Ложные тревоги дорогие (спам) |
| Recall | TP/(TP+FN) | Пропуск опасен (болезнь) |
| F1-score | 2·P·R/(P+R) | Баланс P и R |
| AUC-ROC | Площадь под ROC | Общая качество ранжирования |

### ROC-AUC

ROC кривая: TPR vs FPR при разных порогах.
AUC = 0.5 → случайная модель, AUC = 1.0 → идеальная модель.

### PR-AUC (Precision-Recall)

Лучше ROC-AUC при сильно несбалансированных классах.

---

## 📐 Метрики регрессии

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| MAE | (1/N)·Σ|ŷ-y| | Средняя абс. ошибка, устойчив к выбросам |
| MSE | (1/N)·Σ(ŷ-y)² | Квадратичная, штрафует выбросы сильнее |
| RMSE | √MSE | В тех же единицах что и y |
| R² | 1 - SS_res/SS_tot | Доля объяснённой дисперсии [0, 1] |
| MAPE | (1/N)·Σ|y-ŷ|/y·100% | Относительная ошибка в % |

---

## 💻 Реализация

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)

def compute_all_metrics(y_true, y_pred, y_prob):
    """Вычисляет все метрики классификации."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    return {'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1, 'auc': auc}


def plot_roc_pr_curves(y_true, y_prob, ax1, ax2):
    """Рисует ROC и PR кривые."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax1.plot([0,1], [0,1], 'k--')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curve'); ax1.legend()
    
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax2.plot(rec, prec, label=f'AP = {ap:.3f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('PR Curve'); ax2.legend()


# Пример использования
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = compute_all_metrics(y_test, y_pred, y_prob)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_roc_pr_curves(y_test, y_prob, axes[0], axes[1])
plt.tight_layout(); plt.show()
```

---

## 🔑 Ключевые выводы

1. **Accuracy** вводит в заблуждение при несбалансированных классах
2. **Precision**: из всех предсказанных позитивных, сколько реально позитивных
3. **Recall**: из всех реальных позитивных, сколько мы нашли
4. **F1-score**: гармоническое среднее P и R (штрафует дисбаланс)
5. **AUC-ROC**: качество ранжирования, не зависит от порога
6. **PR-AUC**: лучше при сильном дисбалансе классов

---

## ⚡ Практические задания

1. Сравните модели по 5 метрикам на несбалансированном датасете
2. Найдите оптимальный порог для F1 через кривую F1 vs threshold
3. Объясните почему AUC=0.95 может соответствовать Recall=0.3

---

**[← Урок 03: Логистическая регрессия](lesson-03-logistic-regression.md)** | **[Урок 05: Деревья решений →](lesson-05-decision-trees.md)**
