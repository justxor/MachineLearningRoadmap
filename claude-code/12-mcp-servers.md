# 12. Подключение MCP-серверов

## Способы добавления

### CLI-команда

```bash
claude mcp add filesystem \
  --command "npx -y @modelcontextprotocol/server-filesystem /Users/me/code"
```

Для всех своих проектов добавь флаг `--scope user`. Для текущего проекта — `--scope project` (пишется в `.mcp.json`, коммитится в git).

### Ручная правка конфига

`~/.claude.json` (или `.mcp.json` в корне проекта):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/code"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://readonly@localhost:5432/staging"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

## Полезные серверы

### Filesystem

Чтение/запись в разрешённые папки (для работы с файлами вне репо).

### Postgres / SQLite

```
> Посчитай пользователей, регистрировавшихся за последнюю неделю, сгруппируй по источнику.
```

Агент исполнит SELECT и вернёт результат. Важно: давай read-only креды.

### GitHub

Чтение issues, PR, файлов, создание комментариев.

### Playwright / Puppeteer

Агент может открывать страницы, делать скриншоты, вводить формы. Полезно для e2e-тестирования и визуальных багов.

### Sentry

Агент видит трейсы, breadcrumbs, частоту ошибок, и может сам найти файл в репо.

### Brave / DuckDuckGo Search

Поиск в интернете (доки, ansверы по библиотекам).

### Context7

Актуальная документация библиотек (решает проблему устаревших знаний модели).

## /mcp в сессии

Покажет статус всех серверов: запущены ли, какие инструменты дают, есть ли ошибки.

## Инструменты сервера в разрешениях

```json
{
  "permissions": {
    "allow": [
      "mcp__postgres__query",
      "mcp__github__get_pull_request"
    ],
    "deny": [
      "mcp__github__merge_pull_request"
    ]
  }
}
```

Формат: `mcp__<server>__<tool>`.

## Отладка сервера

- `claude mcp list` — все зарегистрированные
- `claude mcp get <name>` — конкретный сервер
- `npx @modelcontextprotocol/inspector <command>` — интерактивный инспектор
- Логи: `~/.cache/claude/mcp/<server>/`

## Не делай так

- Не подключай прод БД с правами write
- Не используй серверы с непроверенным исходным кодом
- Не храни токены в .mcp.json (используй подстановку env `${VAR}`)
- Не добавляй 10+ серверов «на всякий случай» — контекст съедается

## Практика

1. Добавь filesystem MCP для своей папки `~/notes` в user scope.
2. Подними локальный Postgres, подключи через server-postgres с read-only юзером.
3. Попроси агента объяснить схему БД и найти «странные» таблицы.
4. Добавь GitHub MCP, попроси резюмировать открытые issues.
5. Подключи Playwright MCP, попроси сделать скриншот своего локального дев-сервера.
6. Настрой права так, чтобы `mcp__postgres__query` был allow, а остальные спрашивали.
7. Создай `.mcp.json` в репо с 1–2 серверами для команды.
8. Напиши «diagnose\`-промпт: проверь все MCP-серверы, отчитайся, что работает.

## Чек-лист

- [ ] Добавлен filesystem MCP
- [ ] Подключён Postgres/SQLite с read-only
- [ ] GitHub MCP настроен с токеном из env
- [ ] Права для MCP-инструментов в settings.json
- [ ] Использовал inspector для отладки

---

← [11. MCP основы](11-mcp-basics.md) | [13. Кастомные slash-команды →](13-custom-commands.md)
