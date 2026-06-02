# Урок 14: t-SNE и UMAP — нелинейная визуализация

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 13](lesson-13-pca.md) | [Урок 15 →](lesson-15-final-project.md)

---

## 📐 Теория

### Зачем нелинейное снижение размерности?

PCA линейный — не может "развернуть" многообразия (Swiss Roll, спирали, кластеры). t-SNE и UMAP решают это через нелинейные отображения.

### t-SNE (t-Distributed Stochastic Neighbor Embedding, 2008)

**Идея:** сохранить локальную структуру данных при проекции в 2D/3D.

**Алгоритм:**
1. В высокой размерности: строим вероятностное распределение P (Gaussian) — близкие точки имеют высокую совместную вероятность pᵢⱼ
2. В низкой размерности: строим Q (t-распределение с 1 степенью свободы) — широкие хвосты предотвращают "crowding"
3. Минимизируем KL дивергенцию: KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

**Perplexity** ≈ эффективное число соседей (обычно 5-50).

**Ограничения:**
- Не сохраняет глобальную структуру (межкластерные расстояния бессмысленны!)
- Медленный O(N²) или O(N log N) с Barnes-Hut
- Не детерминированный (random_state!)
- Нельзя применить к новым точкам

### UMAP (Uniform Manifold Approximation and Projection, 2018)

**Теоретическая основа:** топологический анализ данных + алгебра Ри.

**Практические преимущества перед t-SNE:**
- В 10-100x быстрее
- Лучше сохраняет глобальную структуру
- Можно применить к новым точкам (transform())
- Масштабируется до больших датасетов

**Ключевые параметры:**
- **n_neighbors:** локальная vs глобальная структура (малое = локальная, большое = глобальная)
- **min_dist:** плотность кластеров (малое = компактные, большое = размытые)

---

## 💻 Практика

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap  # pip install umap-learn
import time

# === Swiss Roll: сравнение PCA vs t-SNE vs UMAP ===
X_swiss, color = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

fig = plt.figure(figsize=(16, 4))

# 3D оригинал
ax = fig.add_subplot(141, projection='3d')
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color, cmap='viridis', s=5)
ax.set_title('Swiss Roll (3D)')

from sklearn.decomposition import PCA
methods = [
    ('PCA', PCA(n_components=2)),
    ('t-SNE', TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)),
    ('UMAP', umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42))
]

for i, (name, method) in enumerate(methods):
    ax = fig.add_subplot(1, 4, i+2)
    t = time.time()
    X_2d = method.fit_transform(X_swiss)
    elapsed = time.time() - t
    
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title(f'{name} ({elapsed:.1f}s)')

plt.tight_layout(); plt.show()


# === MNIST digits: t-SNE и UMAP ===
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Предварительное снижение через PCA (ускоряет t-SNE)
X_pca = PCA(n_components=50).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=0)
X_tsne = tsne.fit_transform(X_pca)
scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ax=axes[0])
axes[0].set_title('t-SNE: MNIST digits')

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_pca)
scatter = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ax=axes[1])
axes[1].set_title('UMAP: MNIST digits')

plt.tight_layout(); plt.show()


# === Влияние параметров t-SNE: perplexity ===
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, perp in zip(axes, [5, 15, 30, 100]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500)
    X_2d = tsne.fit_transform(X_pca[:200])  # подмножество для скорости
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y[:200], cmap='tab10', s=10)
    ax.set_title(f'perplexity={perp}')
plt.suptitle('Effect of Perplexity on t-SNE', fontsize=12)
plt.tight_layout(); plt.show()


# === UMAP для новых точек (transform) ===
X_train_umap = X_pca[:1500]
X_test_umap = X_pca[1500:]

reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
reducer.fit(X_train_umap)

# Применяем к тестовым данным
X_test_2d = reducer.transform(X_test_umap)
print(f"UMAP transform: {X_test_umap.shape} -> {X_test_2d.shape}")
```

---

## 🔑 Ключевые выводы

1. **t-SNE** превосходен для локальной кластерной структуры; межкластерные расстояния **бессмысленны**
2. **UMAP** быстрее, масштабируется, поддерживает transform() — предпочтительнее в продакшене
3. **Perplexity/n_neighbors:** больше → более глобальная структура; меньше → более локальная
4. **Всегда PCA сначала** (до 50 компонент) перед t-SNE/UMAP — ускоряет и убирает шум
5. **random_state!** — без фиксации результаты не воспроизводимы

---

## ⚡ Задания

1. Визуализируйте эмбеддинги BERT/GPT через UMAP — какова структура семантического пространства?
2. Сравните t-SNE perplexity=5 vs 50 на датасете с кластерами разного размера
3. UMAP для поиска аномалий: точки далеко от кластеров = потенциальные выбросы

---

**[← Урок 13: PCA](lesson-13-pca.md)** | **[Урок 15: Итоговый проект →](lesson-15-final-project.md)**
