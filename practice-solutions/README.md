# ✅ Разборы Задач

Подробные разборы всех практических задач с объяснениями, альтернативными подходами и частыми ошибками.

> ⚠️ **Не смотри разбор, пока не попробовал решить задачу самостоятельно!**  
> Минимальное время самостоятельной работы: 30–60 минут для простых задач, 2–3 часа для сложных.

---

## 📋 Список разборов

| # | Задача | Разбор |
|---|--------|--------|
| 01 | EDA — Страховые выплаты | [solution-01-eda.md](solution-01-eda.md) |
| 02 | Feature Engineering — E-commerce | [solution-02-features.md](solution-02-features.md) |
| 03 | Preprocessing — HR Analytics | [solution-03-preprocessing.md](solution-03-preprocessing.md) |
| 04 | Visualization — Маркетинг | [solution-04-visualization.md](solution-04-visualization.md) |
| 05 | Model Training — Кредитный скоринг | [solution-05-modeling.md](solution-05-modeling.md) |
| 06 | Evaluation — Сравнение моделей | [solution-06-evaluation.md](solution-06-evaluation.md) |
| 07 | Pipelines — Недвижимость | [solution-07-pipelines.md](solution-07-pipelines.md) |
| 08 | Time Series — Продажи | [solution-08-timeseries.md](solution-08-timeseries.md) |
| 09 | NLP — Анализ отзывов | [solution-09-nlp.md](solution-09-nlp.md) |
| 10 | Финальный проект — Телеком | [solution-10-final.md](solution-10-final.md) |

---

## 🧭 Структура каждого разбора

Каждый разбор содержит:
1. **Полное рабочее решение** — весь код с комментариями
2. **Объяснение ключевых решений** — почему именно так, а не иначе
3. **Типичные ошибки** — что чаще всего делают неправильно
4. **Альтернативные подходы** — другие валидные решения
5. **Выводы и инсайты** — что важно запомнить

---

## ❗ Типичные ошибки по темам

**EDA:** Пропускают анализ пропусков | Не проверяют выбросы | Не интерпретируют графики

**Feature Engineering:** Data leakage в target encoding | Не сдвигают лаговые признаки | Создают признаки без смысловой обоснованности

**Preprocessing:** fit() на всём датасете (leakage) | SMOTE до split | Не добавляют флаги пропусков

**Modeling:** Смотрят только accuracy | Не делают baseline | Тестируют на val несколько раз

**Evaluation:** Путают Precision и Recall | Не учитывают бизнес-контекст при выборе метрики | Не строят calibration plot

**Time Series:** Случайный split вместо хронологического | Rolling features без shift(1) | Обычный KFold вместо Walk-Forward
