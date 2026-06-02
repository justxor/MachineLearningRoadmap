# Урок 10: Стекинг и блендинг

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 09](lesson-09-boosting-libraries.md) | [Урок 11 →](lesson-11-clustering.md)

---

## 📐 Теория

### Blending (взвешенное усреднение)

```
Final = w₁·M₁(x) + w₂·M₂(x) + ... + wₙ·Mₙ(x)
```

Простой, быстрый, но не оптимальный. Работает когда модели diverse.

### Stacking (мета-обучение)

1. Разбить train на K folds
2. Для каждого fold: обучить базовые модели на остальных K-1, предсказать на текущем
3. Получить Out-of-Fold (OOF) предсказания для всего train
4. Обучить мета-модель на OOF предсказаниях (признаки = предсказания базовых)
5. Финальные предсказания: base models на всём train → meta model

**Ключевое:** OOF предотвращает leakage — мета-модель никогда не видит предсказания на тех же данных, на которых обучались базовые.

---

## 💻 Реализация

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Базовые модели
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
]


def stacking(base_models, X_train, y_train, X_test, n_folds=5):
    """Реализация стекинга с OOF предсказаниями."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    train_meta = np.zeros((len(X_train), len(base_models)))
    test_meta = np.zeros((len(X_test), len(base_models)))
    
    for col, (name, model) in enumerate(base_models):
        test_preds = np.zeros((len(X_test), n_folds))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            
            model.fit(X_tr, y_tr)
            train_meta[val_idx, col] = model.predict_proba(X_val)[:, 1]
            test_preds[:, fold] = model.predict_proba(X_test)[:, 1]
        
        test_meta[:, col] = test_preds.mean(axis=1)
        
        oof_auc = roc_auc_score(y_train, train_meta[:, col])
        print(f"{name}: OOF AUC = {oof_auc:.4f}")
    
    return train_meta, test_meta


# Генерируем OOF предсказания
print("=== Base Models OOF Performance ===")
train_meta, test_meta = stacking(base_models, X_train, y_train, X_test)

# Мета-модель
meta_model = LogisticRegression()
meta_model.fit(train_meta, y_train)
final_preds = meta_model.predict_proba(test_meta)[:, 1]

stack_auc = roc_auc_score(y_test, final_preds)
print(f"\n=== Stacking (LogReg meta) AUC: {stack_auc:.4f} ===")

# Сравнение с отдельными моделями
print("\n=== Individual Models (full train) ===")
for name, model in base_models:
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{name}: AUC = {auc:.4f}")

# Простой блендинг
blend = test_meta.mean(axis=1)
blend_auc = roc_auc_score(y_test, blend)
print(f"\nBlending (avg): AUC = {blend_auc:.4f}")
```

---

## 🔑 Ключевые выводы

1. **Diversity** базовых моделей важнее их индивидуального качества
2. **OOF предсказания** — единственно правильный способ избежать leakage
3. **Простая LogReg** как мета-модель часто лучше сложной (не переобучается)
4. **Blending** проще и быстрее, но stacking обычно даёт +0.001-0.005 AUC
5. Stacking наиболее полезен на **Kaggle** и когда данных достаточно

---

## ⚡ Задания

1. Сравните stacking vs blending на 3 датасетах. Когда разница значимая?
2. Попробуйте XGBoost как мета-модель вместо LogReg. Помогает ли?
3. Реализуйте 2-уровневый стекинг (meta of meta) и объясните почему он редко используется

---

**[← Урок 09: Бустинг](lesson-09-boosting-libraries.md)** | **[Урок 11: Кластеризация →](lesson-11-clustering.md)**
