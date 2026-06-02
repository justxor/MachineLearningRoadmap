# 📈 Задача 04 — Визуализация Данных

**Сложность:** ⭐⭐ | **Время:** 60–90 минут
**Разбор:** [solution-04-visualization.md](../practice-solutions/solution-04-visualization.md)

---

## 📋 Контекст

Ты дата-саентист в маркетинговом агентстве. Тебе нужно подготовить набор визуализаций для презентации клиенту о поведении покупателей в интернет-магазине.

---

## 📦 Датасет

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000
dates = pd.date_range('2023-01-01', periods=n, freq='6h')

df = pd.DataFrame({
    'date':         dates,
    'user_age':     np.random.randint(18, 70, n),
    'gender':       np.random.choice(['M', 'F'], n),
    'category':     np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports', 'Food'], n),
    'channel':      np.random.choice(['Organic', 'Paid', 'Email', 'Social'], n, p=[0.3, 0.25, 0.2, 0.25]),
    'revenue':      np.abs(np.random.lognormal(4.5, 1.0, n)),
    'items_count':  np.random.randint(1, 10, n),
    'discount_pct': np.random.choice([0, 5, 10, 15, 20, 25], n, p=[0.5, 0.1, 0.15, 0.1, 0.1, 0.05]),
    'returned':     np.random.binomial(1, 0.08, n),
    'rating':       np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.15, 0.35, 0.35]),
})
```

---

## ✅ Задания

### Часть 1 — Matplotlib: базовые графики

**1.1.** Создай subplot 2×2 с:
- Гистограммой revenue (50 бинов)
- Boxplot revenue по channel
- Bar chart: средний revenue по category
- Pie chart: распределение channel

Добавь заголовки, подписи осей, легенду. Сохрани как PNG.

**1.2.** Создай временной ряд: суммарный revenue по дням (resample по дате). Добавь скользящее среднее за 7 дней (rolling mean). Выдели праздничные периоды другим цветом (январь).

---

### Часть 2 — Seaborn: статистические графики

**2.1.** Создай FacetGrid: распределение revenue отдельно для каждого channel (4 графика в ряд). Добавь вертикальную линию на median.

**2.2.** Heatmap: средний revenue в разрезе category × channel (pivot table). Аннотируй значениями.

**2.3.** Scatter plot: user_age vs revenue, раскрасить по gender. Добавь regression line для каждого гендера отдельно. Что видно?

**2.4.** Violin plot: распределение rating по category. Интерпретируй результат.

---

### Часть 3 — Plotly: интерактивные графики

**3.1.** Создай интерактивный scatter plot: revenue vs items_count, размер точки = discount_pct, цвет = category, hover = date, channel, returned.

**3.2.** Создай интерактивный временной ряд с возможностью выбора периода (rangeselector: 1m, 3m, 6m, YTD).

**3.3.** Sunburst chart: иерархия channel → category → gender с суммой revenue. Что занимает наибольшую долю?

---

### Часть 4 — Дашборд (бонус ⭐⭐)

**4.1.** Создай дашборд 3×2 (matplotlib subplots) с ключевыми KPI:
- Total Revenue (метрика)
- Revenue по месяцам (line chart)
- Топ-3 категории (bar chart)
- Return rate по каналам (bar chart)
- Revenue vs discount (scatter)
- Распределение rating (bar chart)

Оформи профессионально: единый цветовой стиль, размер шрифтов, отступы.

---

## 📝 Требования к ответу

- Все графики должны быть информативны: заголовок, подписи осей, легенда
- Для каждого графика — 1–2 предложения интерпретации
- Финальный дашборд должен рассказывать историю о бизнесе

---

## 💡 Подсказки

<details>
<summary>Подсказка к 1.2 (временной ряд)</summary>

```python
daily = df.set_index('date')['revenue'].resample('D').sum()
rolling = daily.rolling(7).mean()
```
</details>

<details>
<summary>Подсказка к 2.2 (pivot heatmap)</summary>

```python
pivot = df.pivot_table(values='revenue', index='category', columns='channel', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
```
</details>
