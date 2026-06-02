# 🧠 Курс: Алгоритмы Машинного Обучения

> **Полный практический курс по алгоритмам ML** — от линейной регрессии до градиентного бустинга. Каждый урок содержит теорию, математику, код на Python и практические задания.

[![Уроков](https://img.shields.io/badge/Уроков-15-blue.svg)](#)
[![Уровень](https://img.shields.io/badge/Уровень-Базовый--Продвинутый-green.svg)](#)
[![Язык](https://img.shields.io/badge/Язык-Python-yellow.svg)](#)

---

## 📋 О курсе

Этот курс охватывает **классические алгоритмы машинного обучения** — фундамент, без которого невозможно понять глубокое обучение и современные LLM. Вы не просто научитесь вызывать `sklearn`, а поймёте, **почему алгоритмы работают именно так**.

**Что вы получите:**
- Математическое понимание каждого алгоритма
- Реализацию с нуля на Python (без sklearn)
- Практические примеры на реальных данных
- Понимание когда какой алгоритм применять

**Предварительные требования:**
- Python на уровне функций, классов, list comprehension
- Базовая линейная алгебра (матрицы, векторы)
- Основы статистики (среднее, дисперсия, вероятность)

---

## 🗺️ Карта курса

```
Линейные модели → Метрики → Деревья → Ансамбли → Кластеризация → Снижение размерности
     ↓               ↓          ↓          ↓              ↓                  ↓
  Урок 1-3        Урок 4     Урок 5-6   Урок 7-10     Урок 11-12         Урок 13-14
                                                                          Урок 15 (Итог)
```

---

## 📚 Список уроков

| № | Урок | Алгоритмы | Время |
|---|------|-----------|-------|
| 01 | [Линейная регрессия: теория и реализация](lesson-01-linear-regression.md) | Linear Regression, OLS, Gradient Descent | ~3 ч |
| 02 | [Регуляризация: Ridge, Lasso, ElasticNet](lesson-02-regularization.md) | L1/L2 regularization, Ridge, Lasso | ~2 ч |
| 03 | [Логистическая регрессия и классификация](lesson-03-logistic-regression.md) | Logistic Regression, Sigmoid, Cross-Entropy | ~3 ч |
| 04 | [Метрики качества ML моделей](lesson-04-metrics.md) | Accuracy, Precision, Recall, F1, ROC-AUC, MSE | ~2 ч |
| 05 | [Деревья решений](lesson-05-decision-trees.md) | Decision Tree, Gini, Entropy, CART | ~3 ч |
| 06 | [Метод ближайших соседей (KNN)](lesson-06-knn.md) | KNN, Distance metrics, KD-tree | ~2 ч |
| 07 | [Random Forest: ансамбль деревьев](lesson-07-random-forest.md) | Random Forest, Bagging, Feature importance | ~3 ч |
| 08 | [Gradient Boosting: теория](lesson-08-gradient-boosting-theory.md) | Gradient Boosting, Weak learners, Loss functions | ~3 ч |
| 09 | [XGBoost, LightGBM, CatBoost](lesson-09-boosting-libraries.md) | XGBoost, LightGBM, CatBoost, Hyperparameters | ~4 ч |
| 10 | [Стекинг и блендинг](lesson-10-stacking.md) | Stacking, Blending, Meta-learner | ~2 ч |
| 11 | [Кластеризация: K-Means и DBSCAN](lesson-11-clustering.md) | K-Means, DBSCAN, Elbow method, Silhouette | ~3 ч |
| 12 | [Иерархическая кластеризация и GMM](lesson-12-hierarchical-gmm.md) | Hierarchical Clustering, GMM, EM algorithm | ~2 ч |
| 13 | [Метод главных компонент (PCA)](lesson-13-pca.md) | PCA, SVD, Explained variance | ~3 ч |
| 14 | [t-SNE и UMAP: визуализация данных](lesson-14-tsne-umap.md) | t-SNE, UMAP, Manifold learning | ~2 ч |
| 15 | [Итоговый проект: ML Pipeline от А до Я](lesson-15-final-project.md) | Feature engineering, Cross-validation, Pipeline | ~5 ч |

**Итого: ~42 часа практики**

---

## 🚀 Быстрый старт

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost umap-learn
jupyter notebook lesson-01-linear-regression.ipynb
```

---

## 📖 Детальное описание уроков

### Урок 01 — Линейная регрессия
**Файл:** [lesson-01-linear-regression.md](lesson-01-linear-regression.md)

Фундамент всего ML. MSE loss, Normal Equation θ = (XᵀX)⁻¹Xᵀy, Gradient Descent. Реализуем с нуля на NumPy. Объясняем когда аналитическое решение неприменимо.

**Ключевые концепции:** MSE, Normal equation, Learning rate, Batch/SGD/Mini-batch GD, Feature scaling

---

### Урок 02 — Регуляризация
**Файл:** [lesson-02-regularization.md](lesson-02-regularization.md)

Ridge: штраф L2 = λΣwᵢ². Lasso: L1 = λΣ|wᵢ| (обнуляет незначимые признаки). ElasticNet = компромисс. Как выбрать λ через кросс-валидацию.

**Ключевые концепции:** Overfitting, Bias-Variance tradeoff, L1/L2, Cross-validation

---

### Урок 03 — Логистическая регрессия
**Файл:** [lesson-03-logistic-regression.md](lesson-03-logistic-regression.md)

Sigmoid σ(z) = 1/(1+e⁻ᶻ). Binary Cross-Entropy. Softmax для мультикласса.

**Ключевые концепции:** Sigmoid, Cross-entropy, MLE, Decision boundary, Softmax

---

### Урок 04 — Метрики качества
**Файл:** [lesson-04-metrics.md](lesson-04-metrics.md)

Когда Precision важнее Recall и наоборот. AUC-ROC, PR-кривая. Метрики регрессии: MAE/MSE/RMSE/R²/MAPE.

**Ключевые концепции:** Confusion matrix, Precision/Recall, F1, ROC-AUC, PR-AUC

---

### Урок 05 — Деревья решений
**Файл:** [lesson-05-decision-trees.md](lesson-05-decision-trees.md)

Gini G = 1 - Σpᵢ². Entropy H = -Σpᵢlog₂(pᵢ). CART алгоритм рекурсивно. Прунинг против переобучения.

**Ключевые концепции:** Gini, Entropy, Information Gain, CART, Pruning, max_depth

---

### Урок 06 — KNN
**Файл:** [lesson-06-knn.md](lesson-06-knn.md)

Евклид/Манхэттен/Минковский. KD-tree для ускорения. Curse of dimensionality.

**Ключевые концепции:** Distance metrics, KD-tree, Ball tree, Weighted KNN

---

### Урок 07 — Random Forest
**Файл:** [lesson-07-random-forest.md](lesson-07-random-forest.md)

Bagging = Bootstrap + Aggregating. Случайный выбор max_features снижает корреляцию деревьев. OOB error.

**Ключевые концепции:** Bootstrap, Bagging, OOB error, Feature importance, n_estimators

---

### Урок 08 — Gradient Boosting: теория
**Файл:** [lesson-08-gradient-boosting-theory.md](lesson-08-gradient-boosting-theory.md)

Каждое дерево подгоняется под псевдо-остатки = отрицательный градиент loss. Функциональный GD.

**Ключевые концепции:** AdaBoost, Residuals, Functional GD, Shrinkage, n_estimators

---

### Урок 09 — XGBoost, LightGBM, CatBoost
**Файл:** [lesson-09-boosting-libraries.md](lesson-09-boosting-libraries.md)

XGBoost: Level-wise + L1/L2. LightGBM: Leaf-wise, GOSS, EFB. CatBoost: Ordered Boosting, автокатегории.

**Ключевые концепции:** Level-wise vs Leaf-wise, GOSS, EFB, Ordered encoding, Early stopping, Optuna

---

### Урок 10 — Стекинг и блендинг
**Файл:** [lesson-10-stacking.md](lesson-10-stacking.md)

Blending: взвешенное среднее. Stacking: мета-модель на OOF-предсказаниях. Как избежать leakage.

**Ключевые концепции:** Meta-learner, Out-of-fold, Diversity, Leakage prevention

---

### Урок 11 — Кластеризация
**Файл:** [lesson-11-clustering.md](lesson-11-clustering.md)

K-Means: минимизация инерции, Elbow method. DBSCAN: плотностная кластеризация, выбросы = шум.

**Ключевые концепции:** Inertia, K-Means++, Silhouette, eps/min_samples, Core/Border/Noise

---

### Урок 12 — Иерархическая кластеризация и GMM
**Файл:** [lesson-12-hierarchical-gmm.md](lesson-12-hierarchical-gmm.md)

Дендрограмма для выбора числа кластеров. GMM: мягкая принадлежность через EM алгоритм.

**Ключевые концепции:** Dendrogram, Linkage (Ward/Complete/Average), GMM, EM, BIC/AIC

---

### Урок 13 — PCA
**Файл:** [lesson-13-pca.md](lesson-13-pca.md)

Ковариационная матрица. Собственные векторы = главные компоненты. SVD: X = UΣVᵀ.

**Ключевые концепции:** Covariance, Eigenvectors/values, Explained variance, SVD, Whitening

---

### Урок 14 — t-SNE и UMAP
**Файл:** [lesson-14-tsne-umap.md](lesson-14-tsne-umap.md)

t-SNE: KL-дивергенция, локальная структура, медленный. UMAP: быстрее, лучше глобальная структура.

**Ключевые концепции:** Manifold, Perplexity, KL div, UMAP graph, n_neighbors, min_dist

---

### Урок 15 — Итоговый проект
**Файл:** [lesson-15-final-project.md](lesson-15-final-project.md)

Полный ML pipeline: EDA → Feature Engineering → Baseline → Модели → Ансамбль → Kaggle submission.

**Ключевые концепции:** EDA, Feature engineering, Pipeline, StratifiedKFold, Model selection, Ensemble

---

## 🛠️ Инструменты курса

| Инструмент | Применение |
|-----------|-----------|
| NumPy | Линейная алгебра, реализации с нуля |
| Pandas | Загрузка и обработка данных |
| Matplotlib/Seaborn | Визуализация |
| Scikit-learn | Эталонные реализации для сравнения |
| XGBoost/LightGBM/CatBoost | Градиентный бустинг |
| Optuna | Автоподбор гиперпараметров |
| Jupyter Notebook | Интерактивная разработка |

---

## ✅ Чеклист прогресса

- [ ] Урок 01: Линейная регрессия с нуля, понял GD
- [ ] Урок 02: Сравнил Ridge/Lasso, выбрал λ через CV
- [ ] Урок 03: Написал логистическую регрессию, visualized decision boundary
- [ ] Урок 04: Знаю когда какую метрику использовать
- [ ] Урок 05: Реализовал дерево рекурсивно, понял Gini vs Entropy
- [ ] Урок 06: KNN с разными метриками расстояния
- [ ] Урок 07: Random Forest с OOB и feature importance
- [ ] Урок 08: Понял functional GD, реализовал простой GBM
- [ ] Урок 09: XGBoost/LightGBM/CatBoost benchmark
- [ ] Урок 10: Стекинг-ансамбль без leakage
- [ ] Урок 11: Сегментировал клиентов K-Means + DBSCAN
- [ ] Урок 12: Дендрограмма + GMM по BIC
- [ ] Урок 13: PCA через SVD, сжал изображения
- [ ] Урок 14: t-SNE и UMAP визуализация эмбеддингов
- [ ] Урок 15: Полный ML pipeline, Kaggle submission

---

## 🔗 Что дальше

- **[Нейронные сети](../neural-networks/README.md)** — Deep Learning с PyTorch
- **[Data Science](../data-science/README.md)** — Полный цикл DS проекта
- **[Math for ML](../math-for-ml/README.md)** — Углублённая математика

---

*Курс является частью [Machine Learning Roadmap](../README.md)*
