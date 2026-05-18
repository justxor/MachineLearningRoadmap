# 💻 Coding-интервью для ML-инженеров

Что спрашивают на coding-этапе ML-собесов и как готовиться эффективно. Без воды — конкретные задачи с разбором подхода.

## 🎯 Что спрашивают на ML coding

Coding для ML-роли — не классический LeetCode. Это смесь:

1. **Алгоритмы (50%):** массивы, строки, hash maps, two pointers, BFS/DFS. Уровень easy-medium.
2. **SQL (20%):** для DS/DA позиций обязательно. Window functions, CTE, optimization.
3. **ML с нуля (20%):** «реализуйте kNN», «напишите gradient descent», «реализуйте softmax».
4. **Python data manipulation (10%):** pandas/numpy, оптимизация, профилирование.

**Сениорные ML-собесы:** алгоритмы менее важны, больше — design кода, чистота, тестируемость.

---

## 📋 Что обязательно знать (топ-25 задач)

### Arrays & Strings

1. **Two Sum** (hash map, O(n)).
2. **Best Time to Buy and Sell Stock** (one pass).
3. **Move Zeroes** (two pointers).
4. **Container With Most Water** (two pointers).
5. **3Sum** (sort + two pointers).
6. **Longest Substring Without Repeating Characters** (sliding window).
7. **Group Anagrams** (hash map с tuple keys).

### Hash maps & Sets

8. **Valid Anagram** (counter).
9. **Top K Frequent Elements** (heap или bucket sort).
10. **Subarray Sum Equals K** (prefix sum + hash map).

### Two Pointers & Sliding Window

11. **Minimum Window Substring** (sliding window + hash map).
12. **Longest Repeating Character Replacement**.

### Trees

13. **Binary Tree Level Order Traversal** (BFS).
14. **Maximum Depth of Binary Tree** (recursion).
15. **Validate BST** (in-order или min/max bounds).
16. **Lowest Common Ancestor**.

### Graphs

17. **Number of Islands** (DFS/BFS на grid).
18. **Course Schedule** (topological sort, cycle detection).
19. **Clone Graph** (BFS/DFS с hash map).

### Dynamic Programming (basics)

20. **Climbing Stairs** (Fibonacci-style).
21. **House Robber**.
22. **Longest Increasing Subsequence**.
23. **Coin Change**.

### Heap & Priority Queue

24. **K Closest Points to Origin** (heap).
25. **Find Median from Data Stream** (two heaps).

**Темп подготовки:** 3-5 задач в день в течение 4 недель. Сначала разобрать паттерн, потом тренировать.

---

## 🧠 Универсальный подход к LeetCode-задаче

1. **Уточнить.** Range входов, edge cases (empty, single element, duplicates, negative numbers).
2. **Brute force.** Озвучить наивное решение и его сложность. Это бейзлайн.
3. **Оптимизация.** Найти узкое место. Hash map / sorting / two pointers / dp.
4. **Объяснить идею.** Не молчите. «Я думаю использовать sliding window, потому что...»
5. **Закодить.** Чисто, с понятными именами. Тесты в голове по ходу.
6. **Тесты.** Прогоните руками на 1-2 примерах. Edge cases.
7. **Complexity.** Time и space. Объяснить почему.

**Антипаттерны:**
- Молчать и сразу писать → интервьюер не понимает, провал.
- Игнорировать edge cases → баги в коде.
- Не упоминать сложность → выглядит неготово.

---

## 📊 SQL для ML / DS интервью

### Обязательный минимум

```sql
-- JOINs
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;

-- Window functions
SELECT
    user_id,
    order_date,
    amount,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY order_date) as cumulative_spent,
    RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) as order_rank
FROM orders;

-- CTE для читаемости
WITH user_metrics AS (
    SELECT user_id, COUNT(*) as orders, SUM(amount) as total
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
),
ranked AS (
    SELECT *, NTILE(10) OVER (ORDER BY total DESC) as decile
    FROM user_metrics
)
SELECT decile, AVG(orders) as avg_orders, AVG(total) as avg_total
FROM ranked
GROUP BY decile;
```

### Топ-задач для практики

- LeetCode Database problems (топ-50).
- StrataScratch (вопросы из реальных собесов).
- HackerRank SQL.

### Типичные вопросы

1. **N-ое максимальное значение** (window function ROW_NUMBER или DENSE_RANK).
2. **Cumulative metrics** (window SUM).
3. **Retention cohort** (self-join + date_trunc).
4. **Sessionization** (gap-based session detection).
5. **Median** (PERCENTILE_CONT или ручной через window).
6. **Funnel analysis** (multiple joins + COUNT(DISTINCT)).
7. **MoM/YoY growth** (LAG window function).
8. **Top N per group** (window RANK).
9. **Pivot/Unpivot** (CASE WHEN или PIVOT).
10. **Time-series gaps** (generate_series + LEFT JOIN).

### Подводные камни

- **NULL behavior:** COUNT(*) vs COUNT(column), DISTINCT с NULL.
- **JOINs vs UNION:** когда что.
- **Index awareness:** не все запросы могут использовать индекс.
- **EXPLAIN ANALYZE:** уметь читать query plan.

---

## 🤖 ML-задачи с нуля (классика собесов)

### 1. Реализовать kNN

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        # Векторизованное вычисление расстояний
        # dists[i, j] = distance from X[i] to X_train[j]
        dists = np.sqrt(
            ((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        # Top k индексов с наименьшим расстоянием
        k_indices = np.argsort(dists, axis=1)[:, :self.k]
        # Голосование
        predictions = []
        for indices in k_indices:
            labels = self.y_train[indices]
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
```

**Что обсудить:** complexity O(N·M·d) for N queries, M train, d features. Как ускорить: KD-Tree / Ball-Tree / approximate (FAISS).

### 2. Реализовать K-Means

```python
def k_means(X, k, max_iter=100, tol=1e-4):
    n, d = X.shape
    # K-Means++ инициализация (упрощённо)
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()
    
    for iteration in range(max_iter):
        # Assign: каждой точке ближайший центроид
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Update: пересчитать центроиды
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Convergence check
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    
    return labels, centroids
```

**Edge cases:** пустые кластеры (некоторое i может не получить ни одной точки), плохая инициализация (запускать N раз).

### 3. Реализовать linear regression с gradient descent

```python
class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros(d)
        self.b = 0
        
        for epoch in range(self.epochs):
            y_pred = X @ self.W + self.b
            error = y_pred - y
            
            # Градиенты (MSE loss)
            dW = (2/n) * X.T @ error
            db = (2/n) * error.sum()
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
    
    def predict(self, X):
        return X @ self.W + self.b
```

**Что добавить на собесе:** regularization (L1/L2), нормализация фичей, ранний останов, мини-батчи.

### 4. Реализовать softmax и cross-entropy

```python
def softmax(x):
    # Numerical stability: вычитаем max
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def cross_entropy(y_true, y_pred):
    # y_true: one-hot, shape (N, C)
    # y_pred: probabilities, shape (N, C)
    eps = 1e-15  # numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

**Подвох:** numerical stability — вычитайте max в softmax. Иначе exp(1000) = inf.

### 5. Реализовать attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

**Что обсудить:** complexity O(n²·d) — отсюда квадратичная сложность трансформера. KV-cache при autoregressive.

### 6. Реализовать precision/recall/F1

```python
def precision_recall_f1(y_true, y_pred):
    # Бинарная классификация, labels 0 и 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
```

**Расширения:** macro/micro/weighted averaging для multi-class.

### 7. Реализовать train/val split

```python
def train_val_split(X, y, val_size=0.2, stratify=None, seed=42):
    np.random.seed(seed)
    n = len(X)
    
    if stratify is not None:
        # Stratified: сохраняем пропорции классов
        train_idx, val_idx = [], []
        for cls in np.unique(stratify):
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            n_val = int(len(cls_indices) * val_size)
            val_idx.extend(cls_indices[:n_val])
            train_idx.extend(cls_indices[n_val:])
    else:
        # Random
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_val = int(n * val_size)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
```

### 8. Реализовать sigmoid и его производную

```python
def sigmoid(x):
    # Numerical stability через два случая
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

---

## 🐍 Python data manipulation: типовые вопросы

### Pandas

1. **Groupby + agg на нескольких метриках.**

```python
df.groupby('category').agg({
    'price': ['mean', 'std', 'count'],
    'quantity': 'sum'
})
```

2. **Merge: inner/left/outer.**

```python
result = pd.merge(orders, users, on='user_id', how='left', indicator=True)
# indicator=True добавляет колонку _merge показывающую источник
```

3. **Rolling window.**

```python
df['rolling_avg_7d'] = df.groupby('user_id')['value'].rolling(7).mean().reset_index(0, drop=True)
```

4. **Pivot table.**

```python
df.pivot_table(values='sales', index='date', columns='product', aggfunc='sum', fill_value=0)
```

### NumPy

1. **Broadcasting:** «нормализуйте каждую колонку матрицы».

```python
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
```

2. **Boolean indexing:** «выберите все строки где column1 > 0 и column2 < 10».

```python
mask = (X[:, 0] > 0) & (X[:, 1] < 10)
filtered = X[mask]
```

3. **Векторизация:** «замените цикл на векторизованную операцию».

```python
# Плохо
result = np.zeros(n)
for i in range(n):
    result[i] = X[i] ** 2 + 3

# Хорошо
result = X ** 2 + 3
```

### Memory & Performance

1. **Memory-efficient dtypes:**
```python
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
```

2. **Chunking больших файлов:**
```python
chunks = pd.read_csv('huge.csv', chunksize=100_000)
results = [process(chunk) for chunk in chunks]
```

3. **Polars или DuckDB** на больших данных (быстрее pandas в 10-100 раз).

---

## ⏱️ Расписание подготовки на 4 недели

**Неделя 1:** Arrays, strings, hash maps (50 задач). Расписание: 3-5 задач в день.

**Неделя 2:** Trees, graphs, recursion (30 задач). Параллельно — SQL базы (window functions).

**Неделя 3:** DP, heap, продвинутые паттерны (30 задач). Параллельно — ML-задачи с нуля.

**Неделя 4:** Mock-интервью, повторение слабых тем, прогон финального чек-листа.

**Темп:** 1.5-2 часа кодинга в день. Регулярность важнее количества.

---

## ✅ Чек-лист готовности к coding-этапу

- [ ] Решил 100+ задач LeetCode medium
- [ ] Все паттерны (two pointers, sliding window, BFS/DFS, DP) — на автомате
- [ ] 50+ SQL задач разной сложности
- [ ] Могу реализовать с нуля: kNN, K-Means, linear regression, softmax, attention
- [ ] Умею писать тесты к своему коду
- [ ] Время на medium задачу < 30 минут
- [ ] Прошёл 3+ mock coding-интервью

---

## 🎯 Ресурсы

- [LeetCode](https://leetcode.com) — Top Interview 150 plan.
- [NeetCode 150](https://neetcode.io) — структурированный план с YouTube-разборами.
- [StrataScratch](https://stratascratch.com) — SQL и data из реальных собесов.
- [interviewing.io](https://interviewing.io) — анонимные mock с инженерами Big Tech.
- [Pramp](https://pramp.com) — бесплатные mock с пирами.
- [HackerRank — Python](https://www.hackerrank.com/domains/python) — для basic Python.

> 💡 Coding — навык, который тренируется. Не «выучить алгоритмы», а **натренировать решать**. Регулярность важнее интенсивности.
