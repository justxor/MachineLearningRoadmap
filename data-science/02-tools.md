# 02. Инструменты: Python, Jupyter, Git, виртуальные окружения

Рабочее окружение — фундамент всего. Если половину времени вы боретесь с пакетами и ImportError, это не работа с данными.

## Минимум

- Python 3.11+
- Менеджер пакетов: uv (рекомендую) или pip+venv, poetry, conda.
- Jupyter Lab / VS Code + Jupyter расширение.
- Git и GitHub account.
- Make / just для повторяемых команд.

## Базовый setup проекта

```bash
uv init ds-project && cd ds-project
uv add pandas numpy scikit-learn matplotlib seaborn jupyterlab
uv add --dev ruff pytest mypy
git init && git add . && git commit -m "init"
```

## Структура папок (cookiecutter-style)

```
.
├── data/         # raw, interim, processed (в .gitignore)
├── notebooks/    # эксперименты
├── src/          # переиспользуемый код
├── reports/      # графики, отчёты
├── tests/
└── pyproject.toml
```

## Jupyter без боли

- Пишите функции в `src/`, в notebook — только вызовы.
- `%load_ext autoreload` + `%autoreload 2` — изменения в модулях подхватываются без restart.
- nbstripout, чтобы не коммитить выводы ячеек.
- Номеруйте notebooks: `01-eda.ipynb`, `02-features.ipynb`.

## Git workflow

- Ветки по фичам, не лью всё в main.
- Коммиты осмысленными блоками. "wip" — плохой коммит.
- .gitignore: `data/`, `*.pkl`, `.env`, `__pycache__/`.

## Антипаттерны

- «Глобальный» Python без venv — конфликты версий.
- Ноутбук в 1500 строк без рефакторинга.
- Данные в git.

## Практика

1. Соберите чистый проект с uv или poetry.
2. Добавьте `pre-commit` с ruff, nbstripout, end-of-file-fixer.
3. Напишите Makefile с целями `lint`, `test`, `format`.
4. Выведите функции из любого старого notebook в `src/`.
5. Настройте GitHub Actions: прогон тестов на пуш.
6. Напишите README с «как запустить» в 3 команды.

## Чек-лист

- [ ] Умею создавать изолированное окружение.
- [ ] Сбрасываю выводы ячеек перед коммитом.
- [ ] Выношу логику из ноутбука в модули.

[← 01](01-roles.md) • [→ 03. SQL](03-sql.md)
