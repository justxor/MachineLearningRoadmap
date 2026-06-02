# Урок 05: Деревья решений

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 04](lesson-04-metrics.md) | [Урок 06 →](lesson-06-knn.md)

---

## 🎯 Цели урока

- Понять критерии разбиения: Gini и Information Gain
- Реализовать CART алгоритм рекурсивно с нуля
- Разобраться с переобучением и методами его предотвращения
- Научиться интерпретировать дерево

---

## 📐 Теория

### Идея

Дерево разбивает пространство признаков на прямоугольные области через иерархические условия if-else.

### Критерии разбиения

**Gini impurity** (для классификации):
```
Gini(S) = 1 - Σₖ pₖ²
```
Минимум при чистом узле (0.0), максимум при равномерном (0.5 для бинарного).

**Information Gain (Entropy)**:
```
H(S) = -Σₖ pₖ · log₂(pₖ)
IG = H(parent) - Σᵢ (|Sᵢ|/|S|) · H(Sᵢ)
```

**MSE** (для регрессии):
```
MSE(S) = (1/|S|)·Σᵢ(yᵢ - ȳ)²
```

### CART алгоритм

1. Для каждого признака и порога вычислить качество разбиения
2. Выбрать лучшее разбиение
3. Рекурсивно повторить для левого и правого поддеревьев
4. Остановиться при достижении критерия останова (max_depth, min_samples)

### Переобучение и Pruning

- Глубокое дерево запоминает шум → плохая обобщаемость
- **Pre-pruning:** ограничить глубину, минимальное число примеров в листе
- **Post-pruning:** обрезать ветки по критерию (Cost-Complexity Pruning)

---

## 💻 Реализация с нуля

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DecisionNode:
    """Узел дерева решений."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Индекс признака для разбиения
        self.threshold = threshold  # Порог разбиения
        self.left = left            # Левое поддерево (x <= threshold)
        self.right = right          # Правое поддерево (x > threshold)
        self.value = value          # Предсказание (для листового узла)


class DecisionTreeFromScratch:
    """Дерево решений CART с нуля."""
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _gini(self, y):
        n = len(y)
        if n == 0: return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1 - np.sum(probs ** 2)
    
    def _entropy(self, y):
        n = len(y)
        if n == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log2(probs + 1e-15))
    
    def _impurity(self, y):
        return self._gini(y) if self.criterion == 'gini' else self._entropy(y)
    
    def _best_split(self, X, y):
        n, d = X.shape
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        
        parent_impurity = self._impurity(y)
        
        for feature in range(d):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                # Information Gain
                n_l, n_r = left_mask.sum(), right_mask.sum()
                child_impurity = (n_l/n) * self._impurity(y[left_mask]) +                                  (n_r/n) * self._impurity(y[right_mask])
                gain = parent_impurity - child_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Условия останова
        if (self.max_depth is not None and depth >= self.max_depth) or            n_samples < self.min_samples_split or            n_classes == 1:
            # Листовой узел: возвращаем наиболее частый класс
            value = np.bincount(y).argmax()
            return DecisionNode(value=value)
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return DecisionNode(value=np.bincount(y).argmax())
        
        left_mask = X[:, feature] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionNode(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y.astype(int))
        return self
    
    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# === Использование ===
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Наша реализация
tree = DecisionTreeFromScratch(max_depth=4, criterion='gini')
tree.fit(X_train, y_train)
acc = np.mean(tree.predict(X_test) == y_test)
print(f"Accuracy (scratch): {acc:.4f}")

# Sklearn для сравнения
sk_tree = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
sk_tree.fit(X_train, y_train)
print(f"Accuracy (sklearn): {sk_tree.score(X_test, y_test):.4f}")

# Визуализация sklearn дерева
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(sk_tree, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True, ax=ax)
plt.title('Decision Tree (max_depth=4)')
plt.tight_layout(); plt.show()
```

---

## 🔑 Ключевые выводы

1. **Gini** и **Entropy** дают похожие результаты; Gini чуть быстрее (нет log)
2. **CART** — жадный алгоритм, не гарантирует глобально оптимальное дерево
3. **max_depth** — главный регуляризатор дерева
4. **Важность признаков** = суммарное снижение Gini по всем узлам где использован признак
5. **Переобучение:** глубокое дерево на train ≈ 100%, на test — плохо

---

## ⚡ Практические задания

1. Реализуйте дерево решений для задачи регрессии (использовать MSE как критерий)
2. Постройте кривые train/test accuracy vs max_depth. Найдите оптимальную глубину
3. Визуализируйте дерево и интерпретируйте первые несколько splits
4. Сравните Gini vs Entropy: есть ли разница в качестве?

---

**[← Урок 04: Метрики](lesson-04-metrics.md)** | **[Урок 06: KNN →](lesson-06-knn.md)**
