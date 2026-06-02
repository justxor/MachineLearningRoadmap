# Урок 13: Метод главных компонент (PCA)

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 12](lesson-12-hierarchical-gmm.md) | [Урок 14 →](lesson-14-tsne-umap.md)

---

## 📐 Теория

### Мотивация

PCA решает задачи:
- Снижение размерности (d признаков → k << d)
- Визуализация (d → 2 или 3)
- Устранение мультиколлинеарности
- Сжатие данных (Eigenfaces)
- Ускорение обучения ML моделей

### Математика PCA

**Шаг 1:** Центрировать данные: X_c = X - mean(X)

**Шаг 2:** Ковариационная матрица:
```
C = (1/(n-1)) · X_cᵀ · X_c
```

**Шаг 3:** Eigen-decomposition:
```
C = V · Λ · Vᵀ
```
V = матрица собственных векторов (главные компоненты)
Λ = диагональная матрица собственных значений (дисперсия)

**Шаг 4:** Проекция:
```
Z = X_c · V[:, :k]    (k компонент)
```

### SVD реализация (численно стабильная)

```
X_c = U · Σ · Vᵀ
```

Главные компоненты = строки Vᵀ = правые сингулярные векторы.
Дисперсия i-й компоненты = σᵢ² / (n-1).

### Explained Variance Ratio

```
EVR_i = λᵢ / Σⱼ λⱼ
```

Cumulative EVR показывает сколько информации сохраняется при k компонентах.

---

## 💻 Реализация PCA с нуля

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class PCAFromScratch:
    """PCA реализация через SVD."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        n, d = X.shape
        
        # Центрирование
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        
        # SVD: X_c = U @ diag(sigma) @ Vt
        U, sigma, Vt = np.linalg.svd(X_c, full_matrices=False)
        
        # Главные компоненты = строки Vt
        self.components_ = Vt
        
        # Объяснённая дисперсия
        self.explained_variance_ = (sigma ** 2) / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        
        return self
    
    def transform(self, X):
        X_c = X - self.mean_
        return X_c @ self.components_.T
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


# === Пример 1: 2D визуализация MNIST ===
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCAFromScratch(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scree plot
pca_full = PCAFromScratch()
pca_full.fit(X_scaled)

ax = axes[0]
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
ax.plot(cumvar); ax.axhline(0.95, color='r', linestyle='--', label='95%')
ax.set_xlabel('Number of components'); ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('Scree Plot'); ax.legend()

# Сколько компонент нужно для 95%?
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Компонент для 95% дисперсии: {n_95} из {X.shape[1]}")

# 2D проекция
ax = axes[1]
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ax=ax)
ax.set_title(f'PCA 2D (EVR: {pca.explained_variance_ratio_.sum():.2%})')

# === Пример 2: Сжатие изображений (Eigenfaces) ===
ax = axes[2]
pca_img = PCAFromScratch(n_components=50)
X_compressed = pca_img.fit_transform(X_scaled)
X_reconstructed = pca_img.inverse_transform(X_compressed)

# Показываем оригинал и реконструкцию
img_idx = 0
orig = scaler.inverse_transform(X_scaled[img_idx:img_idx+1])[0].reshape(8, 8)
recon = scaler.inverse_transform(X_reconstructed[img_idx:img_idx+1])[0].reshape(8, 8)

ax.imshow(np.hstack([orig, recon]), cmap='gray')
ax.set_title(f'Original vs Reconstructed (50 components)')
ax.axis('off')

plt.tight_layout(); plt.show()


# === PCA для preprocessing перед классификацией ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.pipeline import Pipeline

# Сравниваем: без PCA vs с PCA (разное число компонент)
lr = LogisticRegression(max_iter=1000)
score_no_pca = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy').mean()
print(f"\nWithout PCA: {score_no_pca:.4f}")

for n_comp in [10, 20, 30, 40]:
    pipeline = Pipeline([
        ('pca', SklearnPCA(n_components=n_comp)),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    score = cross_val_score(pipeline, X_scaled, y, cv=5, scoring='accuracy').mean()
    print(f"PCA({n_comp:2d}) + LR: {score:.4f}")
```

---

## 🔑 Ключевые выводы

1. **PCA** находит направления максимальной дисперсии (главные компоненты)
2. **Центрирование** данных обязательно; **масштабирование** нужно если признаки в разных единицах
3. **SVD** численно стабильнее чем eigen-decomposition ковариационной матрицы
4. **Explained Variance Ratio** показывает сколько информации сохраняется
5. **PCA не всегда помогает** классификатору — зависит от задачи

---

## ⚡ Задания

1. Реализуйте Incremental PCA для данных, не помещающихся в памяти
2. Kernel PCA (rbf kernel) vs линейный PCA на make_moons — что лучше для visualization?
3. Eigenfaces: обучите PCA на датасете лиц, найдите главные "лица"

---

**[← Урок 12: GMM](lesson-12-hierarchical-gmm.md)** | **[Урок 14: t-SNE и UMAP →](lesson-14-tsne-umap.md)**
