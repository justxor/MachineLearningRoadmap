# Урок 06: Метод ближайших соседей (KNN)

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 05](lesson-05-decision-trees.md) | [Урок 07 →](lesson-07-random-forest.md)

---

## 🎯 Цели урока

- Понять «ленивое» обучение: KNN не строит модель, а запоминает данные
- Разобраться в метриках расстояния и их влиянии
- Изучить Curse of Dimensionality
- Реализовать KNN и оптимизировать через KD-tree

---

## 📐 Теория

### Идея KNN

KNN — «lazy learner»: обучение = запомнить все данные. Предсказание = найти k ближайших соседей и:
- **Классификация:** голосование (majority vote)
- **Регрессия:** среднее значений соседей

### Метрики расстояния

| Метрика | Формула | Когда |
|---------|---------|-------|
| Евклидово | √Σ(xᵢ-yᵢ)² | Стандарт, непрерывные признаки |
| Манхэттенское | Σ|xᵢ-yᵢ| | Устойчив к выбросам |
| Минковского | (Σ|xᵢ-yᵢ|ᵖ)^(1/p) | Обобщение (p=1: Манхэттен, p=2: Евклид) |
| Косинусное | 1 - (x·y)/(||x||·||y||) | Текст, эмбеддинги |

### Curse of Dimensionality

При увеличении размерности d:
- Объём пространства растёт экспоненциально
- Все точки становятся примерно равноудалены
- KNN теряет смысл при d > ~20-30

### Выбор k

- Малый k: высокая дисперсия, чувствителен к шуму
- Большой k: высокое смещение, медленнее
- Оптимум: кросс-валидацией (обычно √N ≤ k ≤ 30)

### Оптимизация: KD-tree и Ball tree

- **Brute force:** O(N·d) на предсказание → медленно при большом N
- **KD-tree:** O(log N) при малом d (d ≤ 20)
- **Ball tree:** лучше при большом d

---

## 💻 Реализация с нуля

```python
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    """KNN классификатор с нуля."""
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        # Вычислить расстояния до всех тренировочных точек
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        
        # Найти k ближайших
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Голосование
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


# === Визуализация влияния k на decision boundary ===
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_values = [1, 5, 15, 50]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, k in zip(axes, k_values):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    # Decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', s=20)
    ax.set_title(f'k={k}, acc={acc:.3f}')

plt.tight_layout(); plt.show()


# Cross-validation для выбора k
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), 
                              X_train, y_train, cv=5).mean() 
             for k in k_range]

best_k = k_range[np.argmax(cv_scores)]
print(f"Best k: {best_k}, CV score: {max(cv_scores):.4f}")
```

---

## 🔑 Ключевые выводы

1. KNN — **non-parametric lazy learner**: нет фазы обучения, всё при предсказании
2. **Feature scaling** критичен: без него признаки с большими значениями доминируют
3. **Curse of dimensionality** делает KNN неэффективным при d > 20
4. **Выбор k**: малый k → overfit, большой k → underfit. Найти через CV
5. **Вычислительная сложность** предсказания: O(N·d) для brute force

---

## ⚡ Практические задания

1. Классификатор рукописных цифр MNIST. Как влияет PCA перед KNN?
2. Сравните метрики расстояния: Euclidean vs Manhattan на make_moons
3. Weighted KNN: соседи ближе голосуют с большим весом. Реализуйте и сравните
4. Бенчмарк KD-tree vs Brute force: время предсказания vs N и d

---

**[← Урок 05: Деревья решений](lesson-05-decision-trees.md)** | **[Урок 07: Random Forest →](lesson-07-random-forest.md)**
