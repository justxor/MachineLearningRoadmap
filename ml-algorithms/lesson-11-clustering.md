# Урок 11: Кластеризация — K-Means и DBSCAN

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 10](lesson-10-stacking.md) | [Урок 12 →](lesson-12-hierarchical-gmm.md)

---

## 📐 Теория

### K-Means

**Алгоритм:**
1. Инициализировать k центроидов случайно (или K-Means++)
2. Назначить каждую точку ближайшему центроиду
3. Пересчитать центроиды (среднее кластера)
4. Повторять до сходимости

**Инерция (Objective):**
```
Inertia = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

**K-Means++ инициализация:**
Выбирает стартовые центроиды с вероятностью ∝ d(x)² (квадрат расстояния до ближайшего центроида) → более стабильная сходимость.

**Elbow method:** строим Inertia vs k, ищем «локоть».

### DBSCAN (Density-Based Spatial Clustering)

Параметры: eps (радиус окрестности), min_samples (минимум точек в окрестности)

**Типы точек:**
- **Core point:** в eps-окрестности ≥ min_samples точек
- **Border point:** в eps-окрестности core point, но сама не core
- **Noise:** не core и не border → выброс!

**Преимущества:**
- Кластеры произвольной формы
- Автоматически определяет выбросы
- Не нужно задавать k

---

## 💻 Реализация K-Means с нуля

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4, init='kmeans++'):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
    
    def _init_centers(self, X):
        n = X.shape[0]
        if self.init == 'random':
            return X[np.random.choice(n, self.k, replace=False)]
        else:  # kmeans++
            centers = [X[np.random.randint(n)]]
            for _ in range(1, self.k):
                dists = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
                probs = dists / dists.sum()
                centers.append(X[np.random.choice(n, p=probs)])
            return np.array(centers)
    
    def fit(self, X):
        self.centers = self._init_centers(X)
        self.inertia_history = []
        
        for _ in range(self.max_iters):
            # Назначить точки кластерам
            dists = np.sqrt(((X[:, None] - self.centers[None])**2).sum(axis=2))
            labels = dists.argmin(axis=1)
            
            # Пересчитать центроиды
            new_centers = np.array([X[labels == k].mean(axis=0) 
                                    if (labels == k).sum() > 0 else self.centers[k]
                                    for k in range(self.k)])
            
            inertia = sum(((X[labels == k] - new_centers[k])**2).sum() 
                         for k in range(self.k) if (labels == k).sum() > 0)
            self.inertia_history.append(inertia)
            
            if np.max(np.abs(new_centers - self.centers)) < self.tol:
                break
            self.centers = new_centers
        
        self.labels_ = labels
        self.inertia_ = self.inertia_history[-1]
        return self
    
    def predict(self, X):
        dists = np.sqrt(((X[:, None] - self.centers[None])**2).sum(axis=2))
        return dists.argmin(axis=1)


# Сегментация клиентов
np.random.seed(42)
# Синтетические данные: age, income, spending_score
data = np.column_stack([
    np.concatenate([np.random.normal(25, 5, 100), np.random.normal(45, 8, 150), 
                    np.random.normal(35, 6, 100)]),
    np.concatenate([np.random.normal(30000, 5000, 100), np.random.normal(70000, 10000, 150),
                    np.random.normal(50000, 8000, 100)]),
    np.concatenate([np.random.normal(70, 10, 100), np.random.normal(30, 8, 150),
                    np.random.normal(55, 12, 100)])
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Elbow method
k_range = range(2, 11)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(k_range, inertias, 'bo-'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[1].plot(k_range, silhouettes, 'go-'); axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
axes[1].set_title('Silhouette Score')

# Финальная кластеризация k=3
best_km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = best_km.fit_predict(X_scaled)
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=30)
axes[2].scatter(best_km.cluster_centers_[:, 0], best_km.cluster_centers_[:, 1],
                c='red', s=200, marker='*', label='Centroids')
axes[2].set_title('K-Means (k=3)'); axes[2].legend()

plt.tight_layout(); plt.show()

# DBSCAN на несферических кластерах
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.15, min_samples=5)
db_labels = dbscan.fit_predict(X_moons)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"DBSCAN: {n_clusters} кластера, {n_noise} выбросов")
print(f"Silhouette (без шума): {silhouette_score(X_moons[db_labels != -1], db_labels[db_labels != -1]):.4f}")
```

---

## 🔑 Ключевые выводы

1. **K-Means** предполагает сферические кластеры равного размера → fails на complex shapes
2. **K-Means++** инициализация значительно стабильнее random
3. **Elbow method + Silhouette** для выбора k: Elbow визуально, Silhouette количественно
4. **DBSCAN** находит кластеры произвольной формы и выбросы автоматически
5. **Scaling обязателен** для обоих алгоритмов!

---

## ⚡ Задания

1. Реализуйте K-Medoids (устойчив к выбросам) и сравните с K-Means
2. Параметры DBSCAN: постройте heatmap Silhouette score vs eps × min_samples
3. Сегментация клиентов: интерпретируйте бизнес-смысл каждого кластера

---

**[← Урок 10: Стекинг](lesson-10-stacking.md)** | **[Урок 12: Иерархическая кластеризация →](lesson-12-hierarchical-gmm.md)**
