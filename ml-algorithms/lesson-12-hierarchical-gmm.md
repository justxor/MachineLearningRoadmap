# Урок 12: Иерархическая кластеризация и GMM

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 11](lesson-11-clustering.md) | [Урок 13 →](lesson-13-pca.md)

---

## 📐 Теория

### Агломеративная иерархическая кластеризация

**Алгоритм:**
1. Каждая точка = отдельный кластер (N кластеров)
2. Объединяем два наиболее близких кластера
3. Повторяем пока не останется 1 кластер
4. Дендрограмма показывает всю историю объединений

**Критерии связи (Linkage):**

| Linkage | Расстояние между кластерами |
|---------|----------------------------|
| Single | min(dist(a,b)) — любые 2 точки |
| Complete | max(dist(a,b)) — самые дальние |
| Average | Среднее всех попарных |
| Ward | Минимизирует SSE при объединении |

Ward — лучший для компактных сферических кластеров.

### GMM (Gaussian Mixture Model)

GMM = "мягкая" кластеризация: каждая точка принадлежит нескольким кластерам с вероятностями.

```
p(x) = Σₖ πₖ · N(x | μₖ, Σₖ)
```

πₖ — веса компонент (Σπₖ = 1), N — гауссиан.

**EM алгоритм:**
- **E-step:** вычислить soft assignments rᵢₖ = P(k | xᵢ)
- **M-step:** обновить параметры (πₖ, μₖ, Σₖ) по soft assignments
- Повторять до сходимости log-likelihood

**Выбор числа компонент:** BIC или AIC (меньше = лучше).

---

## 💻 Реализация

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Данные
X, y_true = make_blobs(n_samples=200, centers=4, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Иерархическая кластеризация ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Дендрограмм для разных linkage
for ax, method in zip(axes[:2], ['ward', 'complete']):
    Z = linkage(X_scaled, method=method)
    dendrogram(Z, truncate_mode='lastp', p=12, ax=ax)
    ax.set_title(f'Dendrogram ({method} linkage)')
    ax.set_xlabel('Samples'); ax.set_ylabel('Distance')

# Финальная кластеризация (Ward, k=4)
Z = linkage(X_scaled, method='ward')
labels = fcluster(Z, t=4, criterion='maxclust')
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=30)
axes[2].set_title('Hierarchical Clustering (Ward, k=4)')

plt.tight_layout(); plt.show()


# === GMM и выбор числа компонент по BIC ===
bic_scores = []
aic_scores = []
k_range = range(1, 9)

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

best_k = k_range[np.argmin(bic_scores)]
print(f"Best k by BIC: {best_k}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_range, bic_scores, 'bo-', label='BIC')
axes[0].plot(k_range, aic_scores, 'rs-', label='AIC')
axes[0].axvline(best_k, color='green', linestyle='--', label=f'Best k={best_k}')
axes[0].set_xlabel('Number of components'); axes[0].set_ylabel('Score')
axes[0].set_title('Model Selection: BIC vs AIC'); axes[0].legend()

# Soft assignments
best_gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
best_gmm.fit(X_scaled)
probs = best_gmm.predict_proba(X_scaled)  # soft assignments
hard_labels = best_gmm.predict(X_scaled)  # hard assignments

ax = axes[1]
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hard_labels, 
                     cmap='viridis', s=30, alpha=0.7)
for k in range(best_k):
    mu = best_gmm.means_[k]
    ax.plot(mu[0], mu[1], 'r*', markersize=15, zorder=5)
ax.set_title(f'GMM Clustering (k={best_k})')

plt.tight_layout(); plt.show()

# Пример мягких принадлежностей
print("\nПример soft assignments (первые 5 точек):")
print(np.round(probs[:5], 3))
```

---

## 🔑 Ключевые выводы

1. **Дендрограмма** = визуализация всей иерархии кластеров без предварительного выбора k
2. **Ward linkage** обычно лучший выбор для компактных кластеров
3. **GMM** = вероятностная кластеризация, гибче K-Means (разные формы и размеры)
4. **BIC предпочтительнее AIC** при выборе числа компонент (сильнее штрафует сложность)
5. **EM сходится** к локальному оптимуму → нужны множественные random restarts

---

## ⚡ Задания

1. Сравните Ward, Single, Complete linkage на датасете с шумом. Какой устойчивее?
2. GMM с ковариационными матрицами 'full' vs 'diag' vs 'spherical' — чем отличаются?
3. Используйте GMM для anomaly detection: низкие p(x) = потенциальные аномалии

---

**[← Урок 11: Кластеризация](lesson-11-clustering.md)** | **[Урок 13: PCA →](lesson-13-pca.md)**
