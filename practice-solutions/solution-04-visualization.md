# ✅ Разбор Задачи 04 — Visualization

**Задание:** [task-04-visualization.md](../practice-tasks/task-04-visualization.md)

---

## 🔑 Ключевые инсайты

1. **Matplotlib subplots** — основа для нестандартных дашбордов
2. **Plotly** проще для интерактивности — меньше кода, лучше результат
3. **FacetGrid** — мощный инструмент для сравнения распределений по группам
4. **Цветовая схема** должна быть единой — не меняй цвета без причины
5. **Интерпретация** — каждый график должен рассказывать историю

---

## 💻 Полное решение

### Часть 1 — Matplotlib subplots

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Генерация данных из задачи
np.random.seed(42)
n = 2000
dates = pd.date_range('2023-01-01', periods=n, freq='6h')
df = pd.DataFrame({
    'date':         dates,
    'user_age':     np.random.randint(18, 70, n),
    'gender':       np.random.choice(['M', 'F'], n),
    'category':     np.random.choice(['Electronics','Clothing','Books','Sports','Food'], n),
    'channel':      np.random.choice(['Organic','Paid','Email','Social'], n, p=[0.3,0.25,0.2,0.25]),
    'revenue':      np.abs(np.random.lognormal(4.5, 1.0, n)),
    'items_count':  np.random.randint(1, 10, n),
    'discount_pct': np.random.choice([0,5,10,15,20,25], n, p=[0.5,0.1,0.15,0.1,0.1,0.05]),
    'returned':     np.random.binomial(1, 0.08, n),
    'rating':       np.random.choice([1,2,3,4,5], n, p=[0.05,0.1,0.15,0.35,0.35]),
})

# 1.1 Subplot 2x2
COLORS = {'Organic': '#2ecc71', 'Paid': '#e74c3c', 'Email': '#3498db', 'Social': '#f39c12'}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Dashboard', fontsize=16, fontweight='bold', y=1.02)

# Гистограмма revenue
axes[0, 0].hist(df['revenue'], bins=50, color='steelblue', edgecolor='none', alpha=0.8)
axes[0, 0].set_title('Revenue Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Revenue (₽)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].axvline(df['revenue'].median(), color='red', linestyle='--', label=f'Median={df["revenue"].median():.0f}')
axes[0, 0].legend()

# Boxplot revenue по channel
channel_data = [df[df['channel']==c]['revenue'].values for c in df['channel'].unique()]
bp = axes[0, 1].boxplot(channel_data, labels=df['channel'].unique(), patch_artist=True,
                         showfliers=False)
for patch, color in zip(bp['boxes'], COLORS.values()):
    patch.set_facecolor(color)
axes[0, 1].set_title('Revenue by Channel', fontweight='bold')
axes[0, 1].set_ylabel('Revenue (₽)')

# Bar chart средний revenue по категории
cat_revenue = df.groupby('category')['revenue'].mean().sort_values(ascending=True)
bars = axes[1, 0].barh(cat_revenue.index, cat_revenue.values, color='#3498db', alpha=0.8)
for bar, val in zip(bars, cat_revenue.values):
    axes[1, 0].text(val + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}₽', va='center', fontsize=9)
axes[1, 0].set_title('Avg Revenue by Category', fontweight='bold')
axes[1, 0].set_xlabel('Avg Revenue (₽)')

# Pie chart channel
channel_counts = df['channel'].value_counts()
axes[1, 1].pie(channel_counts.values, labels=channel_counts.index,
               colors=list(COLORS.values()), autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Channel Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('dashboard_part1.png', dpi=150, bbox_inches='tight')
plt.show()

# 1.2 Временной ряд с rolling mean
daily = df.set_index('date')['revenue'].resample('D').sum()
rolling7 = daily.rolling(7).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily.index, daily.values, alpha=0.4, color='steelblue', linewidth=1, label='Daily Revenue')
ax.plot(rolling7.index, rolling7.values, color='red', linewidth=2, label='7-day Rolling Mean')

# Выделить январь
jan_mask = (daily.index.month == 1)
ax.axvspan(daily.index[jan_mask].min(), daily.index[jan_mask].max(),
           alpha=0.1, color='yellow', label='January (Holiday Period)')
ax.set_title('Daily Revenue with Rolling Mean', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue (₽)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Часть 2 — Seaborn

```python
# 2.2 Heatmap pivot
pivot = df.pivot_table(values='revenue', index='category', columns='channel', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Avg Revenue (₽)'})
ax.set_title('Average Revenue: Category × Channel', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
# ИНТЕРПРЕТАЦИЯ: Paid + Electronics — наибольший средний чек

# 2.3 Scatter plot age vs revenue по gender
g = sns.lmplot(data=df, x='user_age', y='revenue', hue='gender',
               height=6, aspect=1.5, scatter_kws={'alpha': 0.3, 's': 15},
               line_kws={'linewidth': 2})
g.set_axis_labels('User Age', 'Revenue (₽)')
g.fig.suptitle('Age vs Revenue by Gender', y=1.02, fontsize=13)
plt.show()
```

### Часть 3 — Plotly интерактив

```python
# 3.1 Scatter с hover
fig = px.scatter(
    df, x='items_count', y='revenue',
    color='category', size='discount_pct',
    hover_data=['date', 'channel', 'returned'],
    title='Revenue vs Items Count',
    template='plotly_white',
    opacity=0.6,
)
fig.show()

# 3.3 Sunburst
df_sunburst = df.groupby(['channel', 'category', 'gender'])['revenue'].sum().reset_index()
fig = px.sunburst(
    df_sunburst,
    path=['channel', 'category', 'gender'],
    values='revenue',
    title='Revenue Hierarchy: Channel → Category → Gender',
    color='revenue',
    color_continuous_scale='RdYlGn',
)
fig.show()
```

---

## 🐛 Типичные ошибки

1. **Нет подписей осей** — график без подписей непонятен вне контекста кода
2. **Слишком много цветов** — больше 6–7 цветов на одном графике = визуальный шум
3. **Не интерпретировать** — визуализация без вывода не несёт ценности
4. **Pie chart для сравнения** — человек плохо сравнивает углы. Используй bar chart

---

## 📌 Ключевые выводы

- **Paid канал + Electronics** — наивысший средний чек
- **Январь** — потенциально праздничный сезон с пиками
- **Возраст** — слабая линейная зависимость с revenue (одинаково для M/F)
- **Organic** самый большой по объёму (30%), но не обязательно лучший по avg revenue
