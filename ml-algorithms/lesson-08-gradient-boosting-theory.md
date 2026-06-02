# Урок 08: Gradient Boosting — теория

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 07](lesson-07-random-forest.md) | [Урок 09 →](lesson-09-boosting-libraries.md)

---

## 🎯 Цели урока

- Понять чем Boosting отличается от Bagging
- Разобраться в концепции функционального градиентного спуска
- Реализовать простой GBM с нуля
- Понять роль learning rate и n_estimators

---

## 📐 Теория

### AdaBoost: идея Boosting

Обучаем деревья последовательно. Каждое следующее дерево фокусируется на примерах, где предыдущие ошиблись (увеличиваем их веса).

### Gradient Boosting: обобщение

Gradient Boosting рассматривает задачу как функциональную оптимизацию:

Ищем F*(x) = argmin E[L(y, F(x))]

На каждом шаге m:
```
1. Вычислить псевдо-остатки (отрицательный градиент):
   rᵢ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)

2. Обучить дерево hₘ(x) на псевдо-остатках

3. Обновить:
   F(x) ← F(x) + η · hₘ(x)
```

где η (eta) — learning rate (shrinkage).

### Псевдо-остатки для разных loss

| Loss | Псевдо-остатки |
|------|----------------|
| MSE | yᵢ - F(xᵢ) (обычные остатки) |
| MAE | sign(yᵢ - F(xᵢ)) |
| Log loss | yᵢ - p̂ᵢ (как в логрегрессии!) |
| Huber | Комбинация MSE и MAE |

### Learning Rate и N Estimators

Trade-off: маленький η → нужно больше деревьев → медленнее, но лучше качество.

Правило: при уменьшении η в 10x → увеличивайте n_estimators в 10x.

---

## 💻 Реализация простого GBM с нуля

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class GradientBoostingRegressor:
    """Простой Gradient Boosting с нуля."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        self.train_losses = []
    
    def _mse_gradient(self, y, y_pred):
        """Псевдо-остатки для MSE loss: просто остатки."""
        return y - y_pred
    
    def fit(self, X, y):
        # Инициализация: среднее (оптимальная константа для MSE)
        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)
        
        for m in range(self.n_estimators):
            # Псевдо-остатки
            residuals = self._mse_gradient(y, F)
            
            # Обучаем дерево на остатках
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Обновляем предсказания
            F += self.learning_rate * tree.predict(X)
            
            # Логируем loss
            loss = mean_squared_error(y, F)
            self.train_losses.append(loss)
        
        return self
    
    def predict(self, X):
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
    
    def staged_predict(self, X):
        """Предсказания после каждого дерева (для анализа)."""
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()


# Загрузка данных
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сравнение learning rates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for lr, color in zip([0.001, 0.01, 0.1, 0.5], ['blue', 'green', 'red', 'orange']):
    gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=lr, max_depth=3)
    gbm.fit(X_train, y_train)
    ax.plot(gbm.train_losses, label=f'lr={lr}', color=color)

ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
ax.set_title('Training Loss vs Learning Rate'); ax.legend()

# Staged predictions для ошибки на тесте
ax = axes[1]
gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3)
gbm.fit(X_train, y_train)

test_scores = [mean_squared_error(y_test, pred) for pred in gbm.staged_predict(X_test)]
ax.plot(test_scores, label='Test MSE')
ax.plot(gbm.train_losses, label='Train MSE')
ax.set_xlabel('Number of trees'); ax.set_ylabel('MSE')
ax.set_title('Train vs Test MSE'); ax.legend()

plt.tight_layout(); plt.show()

# Сравнение с sklearn GBM
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
sk_gbr = SklearnGBR(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
sk_gbr.fit(X_train, y_train)
print(f"Sklearn GBR Test MSE: {mean_squared_error(y_test, sk_gbr.predict(X_test)):.2f}")
print(f"Our GBR Test MSE:     {mean_squared_error(y_test, gbm.predict(X_test)):.2f}")
```

---

## 🔑 Ключевые выводы

1. **Boosting** строит модели последовательно, исправляя ошибки предыдущих
2. **Псевдо-остатки** = отрицательный градиент loss (для MSE это просто residuals)
3. **Learning rate (shrinkage)** ограничивает вклад каждого дерева
4. **Малый lr + много деревьев** = лучше качество, но медленнее
5. **Early stopping** по validation loss предотвращает переобучение

---

## ⚡ Практические задания

1. Реализуйте GBM для классификации (Binary Log Loss, псевдо-остатки: y - p̂)
2. Добавьте Early Stopping в вашу реализацию (по validation loss)
3. Визуализируйте, как меняются предсказания F(x) от итерации к итерации
4. Реализуйте Huber Loss и сравните с MSE на датасете с выбросами

---

**[← Урок 07: Random Forest](lesson-07-random-forest.md)** | **[Урок 09: XGBoost/LightGBM →](lesson-09-boosting-libraries.md)**
