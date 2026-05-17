# 12. Причинно-следственный анализ (causal inference)

A/B — золотой стандарт, но не всегда возможен. Когда рандомизация невозможна, нужны causal-методы.

## Ключевые идеи

- Counterfactual: «что было бы, если...».
- Confounders — возмущающие переменные.
- Selection bias.
- DAG как язык описания мира.

## Методы

- Propensity score matching.
- Inverse probability weighting.
- Difference-in-differences.
- Regression discontinuity.
- Synthetic control.
- Instrumental variables.
- Uplift modelling (T-/S-/X-learners, causal forests).

## Инструменты

- DoWhy.
- EconML.
- CausalML.
- statsmodels для DiD.

## Антипаттерны

- Называть регрессию «prior controls» с любыми фичами causal-анализом.
- Не рисовать DAG, не формулировать assumptions.
- Пропускать sensitivity analysis.

## Практика

1. Нарисуйте DAG для выбранного феномена (отток клиентов).
2. Реализуйте propensity matching на открытых данных.
3. Постройте DiD на двух регионах и интервенции.
4. Обучите T-learner и X-learner. Сравните.
5. Проведите sensitivity analysis.
6. Объясните в README, что делаете и почему.

## Чек-лист

- [ ] Отличаю предиктивную и causal-задачи.
- [ ] Умею выбрать подходящий метод.
- [ ] Понимаю, что мои выводы хрупки к assumptions.

[← 11](11-ab-testing.md) • [→ 13. ML-канвас](13-ml-canvas.md)
