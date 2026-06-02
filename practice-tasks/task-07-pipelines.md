# 🔗 Задача 07 — ML Пайплайны

**Сложность:** ⭐⭐⭐ | **Время:** 90–120 минут
**Разбор:** [solution-07-pipelines.md](../practice-solutions/solution-07-pipelines.md)

---

## 📋 Контекст

Тебе нужно создать воспроизводимый, production-ready пайплайн для предсказания цен на недвижимость. Пайплайн должен корректно обрабатывать новые данные без переобучения на тестовых.

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 3000

df = pd.DataFrame({
    'area_sqm':       np.abs(np.random.normal(70, 30, n)),
    'rooms':          np.random.randint(1, 6, n),
    'floor':          np.random.randint(1, 25, n),
    'total_floors':   np.random.randint(1, 25, n),
    'build_year':     np.random.randint(1960, 2024, n),
    'district':       np.random.choice(['Center', 'North', 'South', 'East', 'West'], n),
    'condition':      np.random.choice(['New', 'Good', 'Fair', 'Poor'], n, p=[0.2, 0.4, 0.3, 0.1]),
    'has_parking':    np.random.binomial(1, 0.4, n),
    'distance_metro': np.abs(np.random.exponential(1.5, n)),
    'price_million':  np.abs(np.random.normal(8, 4, n)),
})

# Добавляем проблемы
df.loc[np.random.choice(n, 100), 'area_sqm'] = np.nan
df.loc[np.random.choice(n, 80), 'build_year'] = np.nan
df.loc[np.random.choice(n, 60), 'district'] = np.nan
df.loc[5, 'floor'] = 99    # выброс
df.loc[10, 'area_sqm'] = 0  # ошибка
```

---

## ✅ Задания

### Часть 1 — Простой Pipeline

**1.1.** Создай sklearn Pipeline с:
- Импутацией медианой для числовых признаков
- Импутацией модой для категориальных
- StandardScaler для числовых
- OneHotEncoder для категориальных
- Ridge регрессией

Обучи на train, предскажи на test. Выведи RMSE и R².

**1.2.** Докажи отсутствие data leakage: выведи параметры scaler (mean_, scale_), они должны быть посчитаны ТОЛЬКО по train. Как это проверить?

**1.3.** Сохрани пайплайн через joblib. Загрузи обратно и проверь что предсказания идентичны.

---

### Часть 2 — Кастомные трансформеры

**2.1.** Создай кастомный трансформер OutlierClipper:
- Принимает multiplier (IQR * multiplier)
- В fit() вычисляет границы по train
- В transform() клиппирует значения
- Должен работать внутри Pipeline

**2.2.** Создай кастомный трансформер AgeCalculator:
- Принимает колонку build_year
- Создаёт новую колонку building_age = 2024 - build_year
- Удаляет исходную build_year

**2.3.** Создай трансформер FloorRatioAdder:
- Добавляет признак floor_ratio = floor / total_floors
- Добавляет is_top_floor = (floor == total_floors).astype(int)

Включи все три трансформера в финальный Pipeline.

---

### Часть 3 — ColumnTransformer и тюнинг

**3.1.** Используй ColumnTransformer для разных типов колонок. Явно укажи, какие колонки числовые, какие категориальные.

**3.2.** Проведи GridSearchCV на Pipeline. Используй нотацию step__param для параметров:
- Числовые импутация: strategy = ['mean', 'median']
- OneHotEncoder: drop = [None, 'first']
- Ridge: alpha = [0.1, 1.0, 10.0, 100.0]

**3.3.** Выведи лучшие параметры и их значения. Сравни с базовым пайплайном.

---

### Часть 4 — Обработка новых данных (бонус ⭐⭐)

**4.1.** Симулируй production сценарий: создай 10 новых объектов с пропусками, передай в обученный пайплайн. Убедись что предсказания корректны.

**4.2.** Что произойдёт если в production появится новая категория в district (например, 'Airport')? Обработай это в пайплайне (handle_unknown='ignore' vs 'infrequent_if_exist').

**4.3. (Бонус)** Реализуй версионирование пайплайна:
- Сохраняй с метаданными: версия, дата, метрики, список фичей
- Загружай с проверкой версии
- Предупреждай если список фичей изменился
