# 01. Установка и первый запуск

## Что устанавливаем

Claude Code распространяется как npm-пакет. Поддерживаются macOS, Linux, WSL2. Нативный Windows — через WSL.

## Установка

```bash
npm install -g @anthropic-ai/claude-code
```

Проверка:

```bash
claude --version
```

## Аутентификация

Два варианта:

1. **Подписка Claude Pro/Max** — `claude` запускает OAuth-логин в браузере. Лимиты считаются по подписке, ключ API не нужен.
2. **API-ключ Anthropic** — переменная окружения:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Для команды — храните ключи в `~/.zshrc`/`~/.bashrc` или менеджере секретов (1Password CLI, doppler, direnv).

## Первый запуск

Зайди в корень любого репозитория и запусти:

```bash
cd ~/code/my-project
claude
```

Откроется интерактивный REPL. Введи:

```
> Объясни, что делает этот репозиторий. Прочитай README и основные модули.
```

Агент сам прочитает файлы, построит карту и ответит. Не пиши пути вручную — он найдёт сам.

## Ключевые флаги CLI

| Флаг | Назначение |
|------|-----------|
| `-p "prompt"` | Одноразовый запуск без REPL (для скриптов) |
| `--model sonnet` / `opus` | Выбор модели |
| `--resume` | Продолжить последнюю сессию |
| `--continue` | Продолжить с последнего сообщения |
| `--dangerously-skip-permissions` | Отключить подтверждения (только в sandbox/CI) |
| `--output-format json` | Структурированный вывод для парсинга |

## Конфиг пользователя

Глобальные настройки: `~/.claude/settings.json`.

Пример:

```json
{
  "model": "claude-sonnet-4-5",
  "theme": "dark",
  "permissions": {
    "allow": ["Bash(npm test)", "Bash(git status)", "Read", "Write"],
    "deny": ["Bash(rm -rf*)"]
  }
}
```

Проектный конфиг — `.claude/settings.json` в корне репо (коммитится в git, общий для команды).

Локальный override — `.claude/settings.local.json` (в `.gitignore`).

## Практика

1. Установи `claude`, проверь версию.
2. Залогинься (OAuth или API-ключ).
3. Запусти агента в существующем репозитории. Попроси: «Сделай карту архитектуры в 10 строк».
4. Запусти headless: `claude -p "посчитай количество TODO в коде" --output-format json`. Распарси результат `jq`.
5. Создай `~/.claude/settings.json` с deny на `rm -rf` и `git push --force`.
6. Сравни ответы `--model sonnet` и `--model opus` на одной задаче — где разница заметна.
7. Запусти `claude --resume` и проверь, что контекст сохранился.
8. Добавь alias `alias cc="claude"` в shell-конфиг.

## Чек-лист

- [ ] CLI установлен, версия выводится
- [ ] Аутентификация работает
- [ ] Запущен агент в реальном репозитории
- [ ] Headless-режим протестирован
- [ ] Глобальный `settings.json` создан с deny-правилами

---

← [README](README.md) | [02. CLAUDE.md и контекст проекта →](02-claude-md.md)
