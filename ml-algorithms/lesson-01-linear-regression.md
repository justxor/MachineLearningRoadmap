# Урок 01: Линейная регрессия — теория и реализация

**Курс:** [Алгоритмы Машинного Обучения](README.md) | **Следующий урок:** [Урок 02 →](lesson-02-regularization.md)

---

## 🎯 Цели урока

- Понять математику линейной регрессии: что такое гиперплоскость и почему MSE
- Вывести аналитическое решение (Normal Equation) и понять его ограничения
- Реализовать Gradient Descent с нуля и объяснить роль learning rate
- Научиться диагностировать проблемы: underfitting, overfitting, плохой scaling

---

## 📐 Теория

### Постановка задачи

Дано: N обучающих примеров {(x⁽ⁱ⁾, y⁽ⁱ⁾)}, где x⁽ⁱ⁾ ∈ ℝᵈ, y⁽ⁱ⁾ ∈ ℝ.

Цель: найти функцию f(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θdxd = θᵀx, которая минимизирует ошибку предсказания.

В матричной форме: **ŷ = Xθ**, где X ∈ ℝᴺˣ⁽ᵈ⁺¹⁾ — матрица признаков с добавленным столбцом единиц (bias).

### Функция потерь: MSE

**Mean Squared Error (MSE):**
```
J(θ) = (1/2N) · Σᵢ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
     = (1/2N) · ||Xθ - y||²
```

Почему квадрат? Дифференцируем, штрафуем большие ошибки сильнее, выпуклая функция → единственный минимум.

### Аналитическое решение: Normal Equation

Берём производную ∂J/∂θ = 0:

```
∂J/∂θ = (1/N) · Xᵀ(Xθ - y) = 0
Xᵀ · Xθ = Xᵀy
θ = (XᵀX)⁻¹ · Xᵀy
```

**Когда не работает:**
- XᵀX необратима (мультиколлинеарность, d > N)
- d очень большое (O(d³) для инверсии матрицы)

### Gradient Descent

Итеративная оптимизация: движемся в направлении антиградиента.

```
gradient = ∂J/∂θ = (1/N) · Xᵀ(Xθ - y)

θ ← θ - α · gradient
```

где α — learning rate (шаг обучения).

**Варианты:**
- **Batch GD:** градиент по всем N примерам → стабильно, медленно при большом N
- **Stochastic GD (SGD):** градиент по 1 примеру → шумно, быстро сходится
- **Mini-batch GD:** градиент по B примерам → компромисс, стандарт в DL

---

## 💻 Реализация с нуля

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Линейная регрессия с нуля на NumPy.
    Поддерживает Normal Equation и Gradient Descent.
    """
    
    def __init__(self, method='gd', lr=0.01, n_iters=1000):
        self.method = method   # 'normal' или 'gd'
        self.lr = lr           # learning rate для GD
        self.n_iters = n_iters # кол-во итераций для GD
        self.theta = None
        self.loss_history = []
    
    def _add_bias(self, X):
        """Добавляет столбец единиц (intercept term)."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        X_b = self._add_bias(X)   # X_b: (N, d+1)
        n_samples = X_b.shape[0]
        
        if self.method == 'normal':
            self._fit_normal_equation(X_b, y)
        else:
            self._fit_gradient_descent(X_b, y, n_samples)
        
        return self
    
    def _fit_normal_equation(self, X_b, y):
        """θ = (XᵀX)⁻¹Xᵀy"""
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    
    def _fit_gradient_descent(self, X_b, y, n_samples):
        """Batch Gradient Descent."""
        self.theta = np.zeros(X_b.shape[1])
        
        for i in range(self.n_iters):
            # Предсказание
            y_pred = X_b @ self.theta
            
            # Градиент: (1/N) * Xᵀ(ŷ - y)
            gradient = (1/n_samples) * X_b.T @ (y_pred - y)
            
            # Обновление весов
            self.theta -= self.lr * gradient
            
            # Логируем loss
            loss = np.mean((y_pred - y)**2) / 2
            self.loss_history.append(loss)
    
    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b @ self.theta
    
    def mse(self, X, y):
        return np.mean((self.predict(X) - y)**2)
    
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / ss_tot


# === Пример использования ===

# Генерируем данные
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.ravel() + np.random.randn(100) * 0.5

# Нормализация (важно для GD!)
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std

# Обучение через Normal Equation
model_ne = LinearRegression(method='normal')
model_ne.fit(X_scaled, y)
print(f"Normal Equation - theta: {model_ne.theta}")
print(f"R²: {model_ne.r2_score(X_scaled, y):.4f}")

# Обучение через Gradient Descent
model_gd = LinearRegression(method='gd', lr=0.1, n_iters=500)
model_gd.fit(X_scaled, y)
print(f"Gradient Descent - theta: {model_gd.theta}")
print(f"R²: {model_gd.r2_score(X_scaled, y):.4f}")

# Сравнение с sklearn
from sklearn.linear_model import LinearRegression as SklearnLR
sk_model = SklearnLR()
sk_model.fit(X_scaled, y)
print(f"Sklearn R²: {sk_model.score(X_scaled, y):.4f}")
```

---

## 📊 Визуализация

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Данные + линия регрессии
ax = axes[0]
ax.scatter(X, y, alpha=0.6, label='Data')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_scaled = (X_line - X_mean) / X_std
ax.plot(X_line, model_gd.predict(X_line_scaled), 'r-', lw=2, label='Prediction')
ax.set_xlabel('X'); ax.set_ylabel('y')
ax.set_title('Линейная регрессия'); ax.legend()

# 2. Кривая обучения (loss)
ax = axes[1]
ax.plot(model_gd.loss_history)
ax.set_xlabel('Итерация'); ax.set_ylabel('MSE loss')
ax.set_title('Learning curve (GD)')
ax.set_yscale('log')

# 3. Остатки
ax = axes[2]
residuals = y - model_gd.predict(X_scaled)
ax.hist(residuals, bins=20, edgecolor='black')
ax.set_xlabel('Остаток (y - ŷ)'); ax.set_ylabel('Частота')
ax.set_title('Распределение остатков')

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150)
plt.show()
```

---

## ⚡ Практические задания

### Задание 1: Влияние learning rate
Обучите 3 модели с lr = [0.001, 0.1, 1.0]. Постройте learning curves. Что происходит при слишком большом lr?

### Задание 2: Предсказание цен на жильё
Загрузите датасет Boston Housing (или California Housing). Обучите линейную регрессию. Какие признаки наиболее важны? (Посмотрите на значения theta.)

### Задание 3: Mini-batch GD
Доработайте класс: добавьте параметр batch_size. Реализуйте mini-batch GD. Сравните скорость сходимости с Batch GD.

### Задание 4 (продвинутое): Polynomial features
Создайте признаки X, X², X³ вручную. Обучите линейную регрессию. Объясните, как линейная модель может фитировать нелинейные зависимости.

---

## 🔑 Ключевые выводы

1. **Линейная регрессия** предполагает линейную зависимость между признаками и таргетом.
2. **Normal Equation** даёт точное решение, но O(d³) — непрактично при большом числе признаков.
3. **Gradient Descent** масштабируется, но требует выбора learning rate и нормализации признаков.
4. **Feature scaling** критичен для GD: без него алгоритм сходится медленно (или не сходится).
5. **R²** показывает долю объяснённой дисперсии: 1.0 = идеально, 0.0 = модель не лучше среднего.

---

## 📚 Дополнительные ресурсы

- [Stanford CS229 Lecture Notes — Linear Regression](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [3Blue1Brown — Neural Networks (глава о GD)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Hands-On ML (Géron), Chapter 4

---

**[← Назад к курсу](README.md)** | **[Урок 02: Регуляризация →](lesson-02-regularization.md)**
