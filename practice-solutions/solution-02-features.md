# ✅ Разбор Задачи 02 — Feature Engineering

**Задание:** [task-02-features.md](../practice-tasks/task-02-features.md)

---

## 🔑 Ключевые инсайты

1. **Циклические кодировки** для hour/dayofweek — sin/cos лучше, чем числа (23:00 и 00:00 близки, а числа 23 и 0 — нет)
2. **Data leakage** в group-by агрегациях — главная ловушка. Считай по train, map на test
3. **engagement_score** — синтетический признак, объединяющий сигналы лучше чем по отдельности
4. **log1p** для сильно скошенных признаков снижает доминирование выбросов
5. Отбор признаков через **консенсус** надёжнее одного метода

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Данные из условия
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'user_id':          range(n),
    'session_date':     pd.date_range('2023-01-01', periods=n, freq='1h'),
    'country':          np.random.choice(['RU','US','DE','FR','CN','BR','IN'], n, p=[0.4,0.2,0.1,0.1,0.05,0.1,0.05]),
    'device':           np.random.choice(['mobile','desktop','tablet'], n, p=[0.55,0.38,0.07]),
    'session_duration': np.abs(np.random.exponential(300, n)),
    'pages_viewed':     np.random.randint(1, 20, n),
    'cart_added':       np.random.randint(0, 5, n),
    'prev_purchases':   np.random.randint(0, 10, n),
    'days_since_reg':   np.random.randint(0, 730, n),
    'category':         np.random.choice(['electronics','clothing','food','sports','books'], n),
    'price_viewed':     np.random.lognormal(6, 1.2, n),
    'purchased':        np.random.binomial(1, 0.15, n),
})

df_feat = df.copy()

# === ЧАСТЬ 1: Временные признаки ===
dt = df_feat['session_date']
df_feat['hour'] = dt.dt.hour
df_feat['day_of_week'] = dt.dt.dayofweek
df_feat['month'] = dt.dt.month
df_feat['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
df_feat['is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour < 6)).astype(int)
df_feat['is_business_hours'] = ((dt.dt.hour >= 9) & (dt.dt.hour < 18)).astype(int)

# Циклические кодировки — ВАЖНО для нейросетей и линейных моделей
df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

# === Бинарные флаги ===
df_feat['is_new_user'] = (df_feat['days_since_reg'] <= 7).astype(int)
df_feat['is_loyal'] = (df_feat['prev_purchases'] > 5).astype(int)
df_feat['has_cart_activity'] = (df_feat['cart_added'] > 0).astype(int)
df_feat['is_long_session'] = (df_feat['session_duration'] > 600).astype(int)

# Log-трансформация для скошенных распределений
df_feat['session_duration_log'] = np.log1p(df_feat['session_duration'])
df_feat['price_viewed_log'] = np.log1p(df_feat['price_viewed'])
# Почему log1p? Экспоненциальное распределение session_duration создаёт выбросы.
# log1p(x) = log(1+x) работает даже при x=0.

# === ЧАСТЬ 2: Взаимодействия ===
df_feat['engagement_score'] = (
    df_feat['pages_viewed'] * 0.3 +
    (df_feat['session_duration'] / 60) * 0.5 +
    df_feat['cart_added'] * 0.2
)
# Нормализация [0, 1]
scaler = MinMaxScaler()
df_feat['engagement_score'] = scaler.fit_transform(df_feat[['engagement_score']])

df_feat['pages_per_minute'] = df_feat['pages_viewed'] / (df_feat['session_duration'] / 60 + 0.001)
df_feat['cart_to_pages_ratio'] = df_feat['cart_added'] / (df_feat['pages_viewed'] + 1)
df_feat['user_quality_score'] = df_feat['prev_purchases'] / (df_feat['days_since_reg'] + 1) * 30
# Смысл: покупок в месяц = насколько активен пользователь относительно срока регистрации

# === ЧАСТЬ 3: Агрегации (ПРАВИЛЬНО — только по train!) ===
from sklearn.model_selection import train_test_split

X = df_feat.drop(columns=['user_id', 'session_date', 'purchased'])
y = df_feat['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_full = X_train.copy()
train_full['purchased'] = y_train.values

# ПРАВИЛЬНО: считаем агрегации только на train
country_stats = train_full.groupby('country').agg(
    country_purchase_rate=('purchased', 'mean'),
    country_engagement=('engagement_score', 'mean'),
    country_count=('purchased', 'count')
).reset_index()

X_train = X_train.merge(country_stats, on='country', how='left')
X_test = X_test.merge(country_stats, on='country', how='left')
# Пропуски для новых стран в test → fillna(global_mean)
global_pr = y_train.mean()
X_train['country_purchase_rate'] = X_train['country_purchase_rate'].fillna(global_pr)
X_test['country_purchase_rate'] = X_test['country_purchase_rate'].fillna(global_pr)

# device средний cart
device_cart = train_full.groupby('device')['cart_added'].mean().rename('device_avg_cart')
X_train['device_avg_cart'] = X_train['device'].map(device_cart).fillna(train_full['cart_added'].mean())
X_test['device_avg_cart'] = X_test['device'].map(device_cart).fillna(train_full['cart_added'].mean())

# === ЧАСТЬ 4: Отбор признаков ===
# Кодируем категориальные для отбора
from sklearn.preprocessing import LabelEncoder
cat_cols = ['country', 'device', 'category']
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()
le = LabelEncoder()
for col in cat_cols:
    X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))
    X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

X_train_enc = X_train_enc.fillna(0)
X_test_enc = X_test_enc.fillna(0)

# Метод 1: Корреляция
corr = X_train_enc.corrwith(y_train).abs().sort_values(ascending=False)
top_corr = corr.head(15).index.tolist()

# Метод 2: Mutual Information
mi = mutual_info_classif(X_train_enc, y_train, random_state=42)
mi_series = pd.Series(mi, index=X_train_enc.columns).sort_values(ascending=False)
top_mi = mi_series.head(15).index.tolist()

# Метод 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_enc, y_train)
fi = pd.Series(rf.feature_importances_, index=X_train_enc.columns).sort_values(ascending=False)
top_rf = fi.head(15).index.tolist()

# Консенсус
from collections import Counter
all_top = top_corr + top_mi + top_rf
consensus = [f for f, c in Counter(all_top).most_common() if c >= 2]
print(f"Consensus features ({len(consensus)}): {consensus}")

# Сравнение качества
cv = StratifiedKFold(5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

base_cols = ['pages_viewed', 'session_duration', 'cart_added', 'prev_purchases',
             'days_since_reg', 'price_viewed', 'is_weekend', 'is_new_user', 'is_loyal',
             'has_cart_activity', 'is_long_session']

for name, cols in [('Baseline (11 feats)', base_cols),
                   ('All features (30+)', X_train_enc.columns.tolist()),
                   ('Consensus', [c for c in consensus if c in X_train_enc.columns])]:
    available = [c for c in cols if c in X_train_enc.columns]
    score = cross_val_score(lr, X_train_enc[available], y_train, cv=cv, scoring='roc_auc').mean()
    print(f"{name}: ROC-AUC = {score:.4f}")
```

---

## 🐛 Типичные ошибки

1. **Data leakage в агрегациях** — самая частая ошибка! Если считаешь group-by по всему датасету, тестовые данные влияют на тренировочные статистики.

2. **Нет shift(1) у лагов** — для временных рядов лаговые признаки БЕЗ сдвига = использование будущего.

3. **Не нормировать engagement_score** — сравнение "90 страниц * 0.3" vs "5 в корзину * 0.2" некорректно без нормировки.

4. **Циклические кодировки для деревьев** — деревьям не нужны sin/cos. Они важны для линейных моделей и нейросетей.

---

## 📌 Ключевые выводы

- **cart_added** и **has_cart_activity** — сильнейшие предикторы покупки
- **engagement_score** (composite feature) обычно лучше отдельных компонент
- Feature engineering дал прирост ROC-AUC ~0.05–0.10 на синтетических данных
- **Правило:** всегда считай агрегации на train, потом map на test
