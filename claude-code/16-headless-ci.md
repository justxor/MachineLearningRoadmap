# 16. Headless-режим и CI/CD

## Для чего нужен headless

Claude Code в CI и скриптах без REPL. Один вызов — один ответ — предсказуемый вывод:

```bash
claude -p "<prompt>" --output-format json
```

Результат можно парсить `jq`, фильтровать, ложить в PR/issue.

## Сценарии в CI

### 1. Авто-ревью PR

```yaml
# .github/workflows/claude-review.yml
name: Claude PR review
on:
  pull_request:
    types: [opened, synchronize]
jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - name: Install Claude Code
        run: npm i -g @anthropic-ai/claude-code
      - name: Run review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > /tmp/diff.patch
          claude -p "Сделай критическое ревью этого diff: $(cat /tmp/diff.patch)" \
            --output-format json > /tmp/review.json
          jq -r '.result' /tmp/review.json > /tmp/review.md
          gh pr comment ${{ github.event.pull_request.number }} --body-file /tmp/review.md
```

### 2. Автофикс падающих тестов (nightly)

```bash
claude -p "Запусти npm test. Почини падающие тесты. Создай ветку fix/tests-$(date +%F), открой PR. Не меняй продакшн-код, только тесты" \
  --dangerously-skip-permissions
```

### 3. Обновление зависимостей

```bash
claude -p "Посмотри npm outdated. Обнови minor/patch версии, запусти тесты, открой PR списком изменений"
```

### 4. Ответы на issues

На новый лейбл `needs-triage`:

```bash
claude -p "Прочитай issue #$ISSUE_NUMBER. Найди в коде релевантные файлы. Предложи мини-план фикса в комментарии. Не выполняй."
```

## Структура ответа JSON

```json
{
  "type": "result",
  "subtype": "success",
  "result": "<финальный текст>",
  "session_id": "...",
  "total_cost_usd": 0.034,
  "num_turns": 7,
  "is_error": false
}
```

Проверяй `is_error` и exit code перед использованием `result`.

## Потоковый JSON

`--output-format stream-json` — каждое событие (мысль, вызов инструмента, результат) отдельной строкой. Полезно для обёрток с прогресс-баром.

## Безопасность в CI

- Секреты — только через GitHub Actions secrets
- Не запускай `--dangerously-skip-permissions` без изолированного runner
- Ограничь пермишны в самом workflow (`permissions:`)
- Не давай агенту пушить в main — только в feature-ветки
- Prompt injection из PR (текст от внешних контрибьюторов) — реальная угроза. Для forks выключи автозапуск

## Официальное GitHub Action

Anthropic поддерживает `anthropics/claude-code-action` и `anthropics/claude-code-base-action`. Поддерживает trigger на меншн `@claude` в комментариях PR и issues.

## Стоимость

- Ставь лимит `--max-turns N`
- Используй sonnet вместо opus, где возможно
- Бюджет на workflow: проверяй `total_cost_usd` и падай, если выше порога

## Практика

1. Напиши скрипт, который ревьюит локальный `git diff` и пишет отчёт в `review.md`.
2. Настрой GitHub Actions workflow для авто-ревью PR.
3. Сделай nightly задачу: фикс падающих тестов в отдельных PR.
4. Напиши issue-triage workflow: по лейблу предлагает план фикса.
5. Добавь проверку бюджета: workflow падает при `total_cost_usd > 0.50`.
6. Попробуй stream-json и оберни в CLI-прогресс-бар.
7. Официальный action: подключи ответ на «@claude» в комментариях.
8. Напиши pre-commit хук, который запускает короткое `claude -p` для проверки коммит-мессиджа.

## Чек-лист

- [ ] Headless-запуск с `-p` и JSON output
- [ ] CI workflow с Claude Code
- [ ] Secrets через GitHub Actions secrets
- [ ] Контроль стоимости и max-turns
- [ ] Осознаёшь риск injection из PR/issues

---

← [15. Hooks](15-hooks.md) | [17. Стоимость, токены, контекст →](17-cost-context.md)
