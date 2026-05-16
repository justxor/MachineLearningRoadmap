# Шпаргалка по Claude Code

## Команды CLI

| Команда | Назначение |
|---------|-----------|
| `claude` | REPL |
| `claude -p "..."` | Headless |
| `claude --continue` | Продолжить последнюю сессию |
| `claude --resume` | Выбрать сессию |
| `claude --model sonnet` | Выбор модели |
| `claude --output-format json` | JSON вывод |
| `claude --max-turns 10` | Ограничить итерации |
| `claude --dangerously-skip-permissions` | Yolo-режим (только sandbox) |
| `claude mcp add <name>` | Добавить MCP сервер |
| `claude mcp list` | Список MCP |

## Slash-команды в сессии

| Команда | Что делает |
|----------|------------|
| `/init` | Генерирует CLAUDE.md |
| `/clear` | Очистить контекст |
| `/compact` | Сжать историю |
| `/model` | Переключить модель |
| `/cost` | Стоимость сессии |
| `/permissions` | Редактор прав |
| `/agents` | Subagents |
| `/mcp` | Статус MCP |
| `/review` | Ревью |
| `/help` | Справка |

## Клавиатура

- `Esc` — прервать
- `Esc Esc` — откатить к предыдущему сообщению
- `Shift+Tab` — Normal → Plan → Auto-accept
- `@` — файлыв контекст
- `#` — дописать в проектный CLAUDE.md
- `#!` — дописать в глобальный CLAUDE.md
- `Cmd+V` — скриншот

## Файлы конфига

| Файл | Назначение |
|------|-----------|
| `~/.claude/CLAUDE.md` | Глобальная память |
| `./CLAUDE.md` | Проектная (в git) |
| `./CLAUDE.local.md` | Личная (в .gitignore) |
| `~/.claude/settings.json` | Глобальные настройки |
| `.claude/settings.json` | Проектные (в git) |
| `.claude/settings.local.json` | Личные переопределения |
| `.claude/commands/<name>.md` | Slash-команды |
| `.claude/agents/<name>.md` | Subagents |
| `.claude/hooks/<file>.sh` | Скрипты хуков |
| `.mcp.json` | MCP-серверы проекта |

## Имена инструментов (для permissions)

- `Read`, `Edit`, `Write`, `Glob`, `Grep`
- `Bash(<command pattern>)`
- `WebFetch(domain:<host>)`
- `mcp__<server>__<tool>`

## Шаблон permissions

```json
{
  "permissions": {
    "allow": [
      "Read", "Edit", "Write", "Glob", "Grep",
      "Bash(npm test*)", "Bash(npm run lint*)",
      "Bash(git status)", "Bash(git diff*)", "Bash(git log*)",
      "WebFetch(domain:docs.python.org)"
    ],
    "ask": [
      "Bash(git push*)",
      "Bash(npm install*)"
    ],
    "deny": [
      "Bash(rm -rf*)",
      "Bash(git push --force*)",
      "Bash(git reset --hard*)",
      "Bash(npm publish*)",
      "Read(.env*)",
      "Read(**/secrets/**)"
    ]
  }
}
```

## CLAUDE.md шаблон

```markdown
# Project: <имя>

## Stack
- ...

## Commands
- npm run dev
- npm test
- npm run db:migrate

## Architecture
- src/routes/ — HTTP
- src/services/ — логика
- src/db/ — БД

## Conventions
- Conventional Commits
- Импорты через @/

## Do not
- Не запускай db:reset
- Не правь migrations/applied/

## See also
- docs/architecture.md
```

## Шаблон брифа фичи

```
Цель: ...
Контекст: @файлы, ADR
Критерии приёмки:
- тест X проходит
- логи показывают Y
Ограничения:
- не менять public API
```

## Антипаттерны (коротко)

- Коммитить без чтения диффа
- Одна сессия на всё
- Коммиты «рефактор + фича»
- yolo на рабочей машине
- Прод БД на write
- WebFetch открытый в интернет
- Конфиг не в git

## Рефлексы

- Новая задача → `/clear`
- Сложный рефактор → plan mode
- Баг → сначала падающий тест
- Коммит → `git diff` глазами
- Перед выходом → `/cost`

## Полезные MCP серверы

- `@modelcontextprotocol/server-filesystem`
- `@modelcontextprotocol/server-postgres`
- `@modelcontextprotocol/server-github`
- `@modelcontextprotocol/server-sqlite`
- `@modelcontextprotocol/server-puppeteer` / Playwright
- `@sentry/mcp-server`
- `@upstash/context7-mcp` (актуальные доки)

## Headless идиомы

```bash
# JSON результат
claude -p "<prompt>" --output-format json | jq -r .result

# Лимит итераций
claude -p "<prompt>" --max-turns 5

# На выбранной модели
claude -p "<prompt>" --model sonnet

# С явными permissions
claude -p "<prompt>" --allow "Read,Edit,Bash(npm test*)"
```

---

← [README](README.md)
