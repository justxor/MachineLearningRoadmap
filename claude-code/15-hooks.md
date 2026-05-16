# 15. Hooks и автоматизация

## Идея

Hooks — это внешние команды, которые Claude Code запускает в определённые моменты жизненного цикла (перед вызовом инструмента, после изменения файла, перед коммитом). Это детерминированные правила в отличие от промптов.

Полезно для:

- Обязательного lint/format после каждого изменения
- Авто-запуска тестов после изменения кода
- Блокировки опасных команд
- Логирования вызовов для аудита
- Нотификаций («агент ждёт подтверждения»)

## События

| Событие | Когда срабатывает |
|---------|--------------------|
| `PreToolUse` | Перед вызовом инструмента (можно блокировать) |
| `PostToolUse` | После вызова (лог, обработка результата) |
| `UserPromptSubmit` | При отправке промпта пользователем |
| `Notification` | Нужно внимание человека |
| `Stop` | Агент закончил ответ |
| `SubagentStop` | Subagent закончил |
| `SessionStart` / `SessionEnd` | Начало/конец сессии |

## Конфигурация

`.claude/settings.json` (или `~/.claude/settings.json`):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write $CLAUDE_FILE_PATHS && npx eslint --fix $CLAUDE_FILE_PATHS"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/guard.sh"
          }
        ]
      }
    ]
  }
}
```

Matcher фильтрует инструменты по имени (regex). Hook получает JSON на stdin, может вернуть JSON на stdout для информации или изменения поведения.

## Пример: lint после редактирования

Агент правит файл → hook автоматически запускает prettier+eslint → в репо всегда чистый формат. Не нужно просить.

## Пример: guard для bash

`.claude/hooks/guard.sh`:

```bash
#!/usr/bin/env bash
set -e
input=$(cat)
command=$(echo "$input" | jq -r '.tool_input.command // ""')

if echo "$command" | grep -qE 'rm -rf|git push --force|DROP TABLE|TRUNCATE'; then
  echo '{"decision": "block", "reason": "Dangerous command blocked by guard hook"}'
  exit 0
fi
echo '{}'
```

## Пример: аудит-лог

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "command",
          "command": "jq -c '{ts: now, tool: .tool_name, input: .tool_input}' >> ~/.claude/audit.log"
        }]
      }
    ]
  }
}
```

## Пример: авто-тесты

PostToolUse на Edit/Write в `src/**/*.ts`:

```bash
related_test=$(echo "$CLAUDE_FILE_PATHS" | sed 's/src/tests/;s/.ts/.test.ts/')
if [ -f "$related_test" ]; then
  npx vitest run "$related_test"
fi
```

Агент видит результат теста в ответе и сам вразу реагирует на ошибки.

## Границы

- Hooks работают с твоими правами — осторожно с источником команд
- Не тяжёлые: каждый вызов замедляет работу
- Не пиши в них сложную логику — вынеси в скрипты с тестами
- exit code ≠ 0 в PreToolUse блокирует инструмент — используй осознанно

## Нотификации

```json
{
  "matcher": ".*",
  "hooks": [{
    "type": "command",
    "command": "osascript -e 'display notification \"Claude waiting\" with title \"Claude Code\"'"
  }]
}
```

Без этого агент может безмолвно ждать в фоновом терминале.

## Практика

1. Настрой PostToolUse для авто-format после Edit/Write.
2. Напиши PreToolUse guard для блокировки опасных bash-команд.
3. Сделай аудит-лог всех вызовов в `~/.claude/audit.log`.
4. Добавь Notification hook (macOS / Linux notify-send).
5. На PostToolUse для файлов кода запускай связанный тест (если есть).
6. Проверь, что exit !=0 в PreToolUse блокирует вызов.
7. Добавь SessionStart hook, который печатает git branch и status.
8. Оформи все hooks в `.claude/hooks/` и закоммить.

## Чек-лист

- [ ] Auto-format настроен
- [ ] Guard для опасных команд
- [ ] Аудит-лог пишется
- [ ] Notification работает
- [ ] Не переходишь 100 мс на хук

---

← [14. Subagents](14-subagents.md) | [16. Headless-режим и CI/CD →](16-headless-ci.md)
