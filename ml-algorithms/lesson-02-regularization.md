# Урок 02: Регуляризация — Ridge, Lasso, ElasticNet

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 01](lesson-01-linear-regression.md) | [Урок 03 →](lesson-03-logistic-regression.md)

---

## 🎯 Цели урока

- Понять проблему переобучения через Bias-Variance Tradeoff
- Разобраться как L1 и L2 штрафы ограничивают сложность модели
- Реализовать Ridge, Lasso с нуля
- Научиться выбирать λ через кросс-валидацию

---

## 📐 Теория

### Bias-Variance Tradeoff

Total Error = Bias² + Variance + Irreducible Noise

- **High Bias:** модель слишком простая → underfitting
- **High Variance:** модель слишком сложная → overfitting

### Ridge регуляризация (L2)

```
J_Ridge(θ) = (1/2N)||Xθ - y||² + λ||θ||²
θ = (XᵀX + λI)⁻¹ · Xᵀy
```

Веса сжимаются к нулю, но не обнуляются.

### Lasso регуляризация (L1)

```
J_Lasso(θ) = (1/2N)||Xθ - y||² + λ·Σⱼ|θⱼ|
```

Многие веса обнуляются → автоматический отбор признаков (sparse solution).

### ElasticNet (L1 + L2)

```
J(θ) = MSE + λ·(α·L1 + (1-α)/2·L2)
```

α=1: Lasso, α=0: Ridge. Хорош при группах коррелированных признаков.

---

## 💻 Реализация Ridge с нуля

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        n, d = X.shape
        X_b = np.c_[np.ones(n), X]
        I = np.eye(d + 1)
        I[0, 0] = 0  # не регуляризуем intercept
        self.theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        return self
    
    def predict(self, X):
        return np.c_[np.ones(X.shape[0]), X] @ self.theta


# Сравнение Ridge и Lasso на sparse данных
np.random.seed(42)
n, d = 100, 20

# Только 5 признаков важны
true_coef = np.zeros(d)
true_coef[:5] = [3, -2, 1.5, -1, 0.5]

X = np.random.randn(n, d)
y = X @ true_coef + np.random.randn(n) * 0.5

# Regularization paths
alphas = np.logspace(-3, 3, 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, Model, title in zip(axes, [Ridge, Lasso], ['Ridge (L2)', 'Lasso (L1)']):
    coefs = [Model(alpha=a).fit(X, y).coef_ for a in alphas]
    ax.plot(np.log10(alphas), coefs)
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('log10(alpha)'); ax.set_ylabel('Coefficient')
    ax.set_title(f'Regularization Path — {title}')

plt.tight_layout(); plt.show()

# Cross-validation для выбора λ
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X, y)
print(f"Best Ridge alpha: {ridge_cv.alpha_:.4f}")

lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X, y)
print(f"Best Lasso alpha: {lasso_cv.alpha_:.4f}")
print(f"Non-zero Lasso coefs: {(lasso_cv.coef_ != 0).sum()}")
```

---

## 🔑 Ключевые выводы

1. **Ridge (L2)** сжимает веса — хорош когда все признаки релевантны
2. **Lasso (L1)** обнуляет незначимые — встроенный feature selection
3. **ElasticNet** = компромисс, хорош при коррелированных признаках
4. **Выбор λ:** через RidgeCV / LassoCV (cross-validation)
5. **Intercept** не регуляризуется

---

## ⚡ Практические задания

1. Постройте regularization path для Ridge и Lasso. Что при α→0 и α→∞?
2. Используйте LassoCV — найдите оптимальное α и число ненулевых коэффициентов
3. Сравните ElasticNet с разными l1_ratio на датасете с коррелированными признаками

---

**[← Урок 01](lesson-01-linear-regression.md)** | **[Урок 03: Логистическая регрессия →](lesson-03-logistic-regression.md)**
