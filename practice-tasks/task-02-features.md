# 🛠️ Задача 02 — Feature Engineering

**Сложность:** ⭐⭐⭐ | **Время:** 90–120 минут
**Разбор:** [solution-02-features.md](../practice-solutions/solution-02-features.md)

---

## 📋 Контекст

Ты работаешь в e-commerce компании. Задача — предсказать, совершит ли пользователь покупку в течение 7 дней после посещения сайта. Датасет содержит логи сессий пользователей.

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'user_id':         range(n),
    'session_date':    pd.date_range('2023-01-01', periods=n, freq='1h'),
    'country':         np.random.choice(['RU', 'US', 'DE', 'FR', 'CN', 'BR', 'IN'], n, p=[0.4,0.2,0.1,0.1,0.05,0.1,0.05]),
    'device':          np.random.choice(['mobile', 'desktop', 'tablet'], n, p=[0.55, 0.38, 0.07]),
    'session_duration': np.abs(np.random.exponential(300, n)),  # секунды
    'pages_viewed':    np.random.randint(1, 20, n),
    'cart_added':      np.random.randint(0, 5, n),
    'prev_purchases':  np.random.randint(0, 10, n),
    'days_since_reg':  np.random.randint(0, 730, n),
    'category':        np.random.choice(['electronics', 'clothing', 'food', 'sports', 'books'], n),
    'price_viewed':    np.random.lognormal(6, 1.2, n),
    'purchased':       np.random.binomial(1, 0.15, n),
})
```

---

## ✅ Задания

### Часть 1 — Базовые трансформации

**1.1.** Создай временные признаки из session_date:
- hour, day_of_week, month, is_weekend, is_night (22:00–06:00), is_business_hours (9:00–18:00)
- Циклические кодировки для hour и day_of_week (sin/cos)

**1.2.** Создай бинарные флаги:
- is_new_user (days_since_reg <= 7)
- is_loyal (prev_purchases > 5)
- has_cart_activity (cart_added > 0)
- is_long_session (session_duration > 600 секунд)

**1.3.** Трансформируй session_duration и price_viewed с помощью log1p. Почему это важно?

---

### Часть 2 — Взаимодействия признаков

**2.1.** Создай признак engagement_score:
- engagement = (pages_viewed * 0.3) + (session_duration / 60 * 0.5) + (cart_added * 0.2)
- Нормализуй его в диапазон [0, 1]

**2.2.** Создай признак conversion_potential:
- pages_per_minute = pages_viewed / (session_duration / 60 + 0.001)
- cart_to_pages_ratio = cart_added / (pages_viewed + 1)

**2.3.** Создай признак user_quality_score = prev_purchases / (days_since_reg + 1) * 30
Что этот признак означает содержательно?

---

### Часть 3 — Агрегации

**3.1.** Для каждой страны (country) создай агрегационные признаки:
- Среднее purchases_rate по стране
- Средний engagement_score по стране
- Количество записей по стране (частота)

Правильно ли вычислять эти агрегации по ВСЕМУ датасету? Когда это проблема?

**3.2.** Создай признак: средний cart_added для данного device типа.

---

### Часть 4 — Отбор признаков (бонус ⭐⭐)

**4.1.** Используй минимум 3 метода отбора признаков (correlation, mutual_info, RandomForest importance). Составь список топ-10 признаков по каждому методу.

**4.2.** Найди консенсус — признаки, вошедшие в топ-15 хотя бы по 2 из 3 методов.

**4.3.** Обучи LogisticRegression на:
- Исходных 11 признаках
- Всех созданных признаках (30+)
- Только консенсусных признаках

Сравни ROC-AUC на кросс-валидации. Вывод?

---

## 📝 Требования к ответу

- Итоговое количество признаков после каждой части
- Объяснение смысла каждого созданного признака
- Сравнение качества модели до и после feature engineering
- Вывод: какие признаки дали наибольший прирост качества

---

## 💡 Подсказки

<details>
<summary>Подсказка к 1.1 (циклические кодировки)</summary>

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```
</details>

<details>
<summary>Подсказка к 3.1 (утечка данных)</summary>

Если считать агрегации по всему датасету, значения из test попадут в train через статистики — это data leakage. В реальных задачах считай только на train, потом map на test.
</details>
