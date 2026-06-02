# 📊 Лучшие Практики и Сниппеты Data Science

Коллекция готовых к использованию сниппетов кода, шаблонов и лучших практик для Data Science проектов. Каждый файл — это практическое руководство с реальным кодом.

## 📋 Содержание

| # | Тема | Описание | Файл |
|---|------|----------|------|
| 01 | 🔍 EDA | Разведочный анализ данных | [eda-snippets.md](eda-snippets.md) |
| 02 | 🛠️ Feature Engineering | Создание и отбор признаков | [feature-engineering.md](feature-engineering.md) |
| 03 | ⚙️ Preprocessing | Предобработка данных | [preprocessing.md](preprocessing.md) |
| 04 | 📈 Visualization | Визуализация данных | [visualization.md](visualization.md) |
| 05 | 🤖 Model Training | Обучение моделей | [model-training.md](model-training.md) |
| 06 | 📏 Evaluation | Оценка качества моделей | [evaluation.md](evaluation.md) |
| 07 | 🔗 Pipelines | ML пайплайны и автоматизация | [pipelines.md](pipelines.md) |
| 08 | ⏱️ Time Series | Временные ряды | [time-series.md](time-series.md) |
| 09 | 📝 NLP | Обработка текста | [nlp-snippets.md](nlp-snippets.md) |
| 10 | 🚀 Deployment | Деплой и продакшн | [deployment.md](deployment.md) |

---

## 🚀 Быстрый старт

```python
# Минимальный DS шаблон — начни с этого
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Воспроизводимость
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

---

## 💡 Топ-10 Лучших Практик

1. **Всегда устанавливай random seed** — воспроизводимость критична
2. **Проверяй утечку данных** — fit только на train, transform на train+test
3. **Используй пайплайны** — объединяй preprocessing и модель
4. **Логируй эксперименты** — MLflow/W&B для трекинга
5. **Валидируй правильно** — стратифицированный K-fold для классификации
6. **Масштабируй признаки** — многие алгоритмы чувствительны к масштабу
7. **Анализируй ошибки** — смотри на конкретные примеры где модель ошибается
8. **Документируй датасет** — описывай источник, дату, версию данных
9. **Следи за дрейфом данных** — распределение в продакшне ≠ трейн
10. **Baseline прежде всего** — простая модель до сложной

---

## 📁 Структура DS Проекта (Template)

```
my-ds-project/
├── data/
│   ├── raw/          # исходные данные (только чтение!)
│   ├── interim/      # промежуточные обработанные
│   └── processed/    # финальные данные для моделей
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-features.ipynb
│   └── 03-modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/         # скрипты загрузки/обработки данных
│   ├── features/     # генерация признаков
│   ├── models/       # обучение и предсказание
│   └── visualization/ # функции для графиков
├── models/           # сохранённые модели
├── reports/
│   └── figures/      # картинки для отчётов
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

---

## 🔧 Полезные однострочники

```python
# Быстрая информация о датасете
def quick_info(df):
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    print(f"\nMissing:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    return df.describe()

# Память датафрейма
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Уникальные значения для всех колонок
df.nunique().sort_values()

# Корреляция с таргетом
df.corr()['target'].abs().sort_values(ascending=False)

# Процент пропусков
(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
```
