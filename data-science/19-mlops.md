# 19. MLOps: трекинг, регистр моделей, CI/CD

MLOps нужен, чтобы проект не разваливался после первой версии. Обязательный минимум, даже в маленькой команде.

## Элементы

- **Трекинг экспериментов.** MLflow, Weights & Biases, ClearML.
- **Регистр моделей.** MLflow Model Registry, BentoML.
- **Данные.** DVC, lakefs.
- **Оркестрация.** Airflow, Prefect, Dagster.
- **Feature store.** Feast.
- **CI/CD.** GitHub Actions / GitLab CI: линт, тесты, обучение на PR.
- **Сервинг.** KServe, BentoML, FastAPI.

## Минимальный пайплайн

1. git push → GitHub Actions: lint, test.
2. Ручной trigger train: выгрузка данных, обучение, лог в MLflow.
3. Сравнение с baseline. Если лучше — регистрация.
4. По тегу «production» — сборка docker image и deploy.

## Репродуцируемость

- Фиксируйте random_state.
- Сохраняйте версию данных (DVC hash).
- Логируйте git commit + image tag в metadata модели.
- Пинуйте версии библиотек (pyproject.lock).

## Антипаттерны

- model.pkl в git.
- Ручное «deploy с ноутбука».
- Отсутствие версия фичей.
- «Один скрипт решает всё».

## Практика

1. Настройте MLflow локально + сервер.
2. Логируйте эксперименты, сравните 5 ранов.
3. Регистрация лучшей модели в Model Registry.
4. Настройте GitHub Actions: lint + tests.
5. Добавьте train-job, которая пишет метрики в PR.
6. Реализуйте DVC для версионирования данных.

## Чек-лист

- [ ] Эксперименты логируются.
- [ ] Модель версионируется.
- [ ] CI работает на PR.

[← 18](18-productionization.md) • [→ 20. Мониторинг](20-monitoring.md)
