# Капстон 1. End-to-end проект по табличным данным

Проект, который покажет работодателю всю цепочку: бизнес-вопрос → данные → модель → сервис.

## Цель

Сделать полный цикл ML-проекта на табличных данных: churn prediction / credit scoring / demand forecasting (выберите одну).

## Шаги

1. **Постановка.** ML-канвас, бизнес-вопрос, метрики.
2. **Данные.** Сбор (из Kaggle / API). Сохранение в raw/processed.
3. **EDA + cleaning.** Отчёт в reports/.
4. **Feature engineering.** Документируйте фичи.
5. **Бейзлайн и модель.** Бустинг с валидацией.
6. **Интерпретация.** SHAP-отчёт.
7. **MLflow.** Логируйте эксперименты.
8. **API.** FastAPI + Docker.
9. **CI.** GitHub Actions: lint, test, train smoke test.
10. **Отчёт.** README + слайды/PDF с выводами.

## Критерии приёмки

- Код воспроизводим (1 команда запуска).
- README объясняет выбор метрики и модели.
- Есть бейзлайн и сравнение.
- Нет утечек (проверено adversarial validation).
- Есть SHAP-отчёт.
- Сервис запускается в docker.
- CI зелёный.

## Структура репозитория

```
.
├── data/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-features.ipynb
│   └── 03-model.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── app.py
├── tests/
├── reports/
├── Dockerfile
├── Makefile
└── README.md
```

## Дедлайн

2–3 недели при занятости вечерами. Пишите коммиты ежедневно.

[← 22](22-communication.md) • [→ Капстон 2](capstone-2-ab.md)
