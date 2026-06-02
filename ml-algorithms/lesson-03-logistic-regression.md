# Урок 03: Логистическая регрессия и классификация

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 02](lesson-02-regularization.md) | [Урок 04 →](lesson-04-metrics.md)

---

## 🎯 Цели урока

- Понять переход от регрессии к классификации через вероятности
- Разобраться в Sigmoid и Binary Cross-Entropy
- Реализовать логистическую регрессию с нуля
- Освоить Softmax для мультиклассовой задачи

---

## 📐 Теория

### От линейной к логистической регрессии

Линейная регрессия даёт ŷ ∈ (-∞, +∞). Для вероятности нужно ŷ ∈ [0, 1].

**Sigmoid функция:**
```
σ(z) = 1 / (1 + e⁻ᶻ),    z = θᵀx
```

Свойства: σ(0) = 0.5, σ(+∞) → 1, σ(-∞) → 0, σ'(z) = σ(z)(1-σ(z))

### Функция потерь: Binary Cross-Entropy

Максимизируем правдоподобие (MLE):
```
J(θ) = -(1/N) · Σᵢ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

**Градиент (удивительно простой!):**
```
∂J/∂θ = (1/N) · Xᵀ(ŷ - y)
```

Такой же как у линейной регрессии — только ŷ считается через σ.

### Decision Boundary

Граница решений: σ(θᵀx) = 0.5 → θᵀx = 0.

Это гиперплоскость в пространстве признаков.

### Мультиклассовая классификация: Softmax

```
softmax(z)ₖ = e^zₖ / Σⱼ e^zⱼ
```

Превращает вектор z в вероятностное распределение. Loss = Categorical Cross-Entropy.

---

## 💻 Реализация с нуля

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    """Логистическая регрессия с нуля."""
    
    def __init__(self, lr=0.1, n_iters=1000, regularization=None, alpha=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization  # 'l2' или None
        self.alpha = alpha
        self.theta = None
        self.losses = []
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n, d = X.shape
        X_b = np.c_[np.ones(n), X]
        self.theta = np.zeros(d + 1)
        
        for _ in range(self.n_iters):
            z = X_b @ self.theta
            y_pred = self.sigmoid(z)
            
            # Binary Cross-Entropy loss
            eps = 1e-15
            loss = -np.mean(y * np.log(y_pred + eps) + (1-y) * np.log(1 - y_pred + eps))
            self.losses.append(loss)
            
            # Gradient
            gradient = X_b.T @ (y_pred - y) / n
            
            # L2 regularization (не для intercept)
            if self.regularization == 'l2':
                reg_term = self.alpha * self.theta.copy()
                reg_term[0] = 0  # intercept
                gradient += reg_term / n
            
            self.theta -= self.lr * gradient
        
        return self
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X_b @ self.theta)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# === Бинарная классификация ===
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_iters=500)
model.fit(X_train, y_train)
print(f"Accuracy: {model.accuracy(X_test, y_test):.4f}")

# Визуализация decision boundary
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=30)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(model, X_train, y_train, axes[0], 'Decision Boundary (Train)')
axes[1].plot(model.losses)
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('BCE Loss')
axes[1].set_title('Training Loss')
plt.tight_layout(); plt.show()
```

---

## 🔑 Ключевые выводы

1. **Sigmoid** отображает линейный выход в вероятность [0, 1]
2. **BCE loss** — правильная функция потерь для бинарной классификации
3. **Градиент** имеет ту же форму что и у линейной регрессии (только ŷ другой)
4. **Decision boundary** — гиперплоскость θᵀx = 0
5. **Softmax** обобщает sigmoid для K классов

---

## ⚡ Практические задания

1. **Детектор спама:** обучите модель на датасете SMS Spam Collection. Метрики: Precision, Recall, F1.
2. **Визуализация:** нарисуйте decision boundary для разных значений learning rate
3. **Порог классификации:** поиграйте с threshold. Как меняются TP/FP при threshold=0.3 vs 0.7?
4. **Мультикласс:** реализуйте Softmax regression на датасете Iris (3 класса)

---

**[← Урок 02: Регуляризация](lesson-02-regularization.md)** | **[Урок 04: Метрики →](lesson-04-metrics.md)**
