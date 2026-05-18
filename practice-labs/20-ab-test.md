# 🧪 Лаба 20: A/B-тест ML-фичи end-to-end 🔴

## Цель

Спроектировать, запустить и проанализировать A/B-тест ML-модели: от гипотезы и расчёта sample size до финального решения. Научиться избегать ловушек (p-hacking, novelty, network effects).

## Компоненты

- Hypothesis doc (что тестируем, почему, какой effect size ожидаем).
- Power analysis для sample size.
- Randomization unit (user, session, request).
- Feature flagging (Unleash, LaunchDarkly, GrowthBook).
- Metric pipeline.
- Анализ и решение.

## Минимальный пайплайн

1. Hypothesis: «Новый ranker увеличит CTR на ≥2%».
2. Power analysis: при baseline CTR 10%, MDE 0.2pp → N=15000 на ветку.
3. AA-тест для валидации рандомизации.
4. Запуск A/B с hash-based split.
5. Базовые метрики: primary (CTR), secondary (revenue), guardrail (latency, error rate).
6. Анализ: t-test/CUPED для снижения variance.
7. Решение: ship / kill / iterate.

## Код: CUPED

```python
import numpy as np
from scipy import stats

def cuped(y, y_pre):
    theta = np.cov(y, y_pre)[0,1] / np.var(y_pre)
    y_adj = y - theta * (y_pre - y_pre.mean())
    return y_adj

y_a = cuped(metrics_a, pre_metrics_a)
y_b = cuped(metrics_b, pre_metrics_b)
t, p = stats.ttest_ind(y_a, y_b, equal_var=False)
```

## Критические вещи

- Pre-registration: метрики и логика решения фиксируются ДО запуска.
- Minimum runtime ≥1 неделя (weekly seasonality).
- Sample ratio mismatch чек (сплит работает?).
- Multiple comparisons — поправка Bonferroni/FDR.
- Heterogeneous treatment effects (HTE).

## Расширения

- Sequential testing (mSPRT, Bayesian).
- Switchback tests для marketplace.
- Cluster randomization.
- Quasi-experiments и DiD.
- Uplift modeling после A/B.

## Критерии приёмки

- [ ] Design doc в репо (hypothesis, metrics, decision rules).
- [ ] Sample size рассчитан.
- [ ] AA-тест проведён.
- [ ] SRM-чек пройден.
- [ ] CUPED/CUPAC применён.
- [ ] Финальный отчёт с интервалами и решением.

## Анти-паттерны

- ❌ Peeking: «посмотрели через день, вроде выигрываем, остановили» (ложный успех).
- ❌ Смена метрики после запуска (cherry-picking).
- ❌ Sample ratio mismatch игнорируется.
- ❌ Решение без guardrail метрик («CTR вырос, прибыль упала»).
- ❌ Игнор novelty effect.

---

[← Назад к Practice Labs](./README.md)
