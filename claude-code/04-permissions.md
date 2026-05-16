# 04. Permissions и безопасность

## Зачем это важно

Агент имеет реальный доступ к файловой системе и shell. Ошибка в промпте или двусмысленная инструкция может стоить ветки, базы или прода. Permissions — это предохранитель.

## Модель разрешений

Каждый инструмент (чтение файла, запись, bash, MCP-вызов) по умолчанию спрашивает подтверждение. Можно разрешить заранее или запретить полностью.

Иерархия (выигрывает более конкретный deny):

1. `~/.claude/settings.json` — глобальные
2. `.claude/settings.json` — проектные (в git)
3. `.claude/settings.local.json` — личные (в .gitignore)

## Синтакс

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit",
      "Write",
      "Bash(npm test*)",
      "Bash(npm run lint*)",
      "Bash(git status)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "WebFetch(domain:docs.python.org)"
    ],
    "deny": [
      "Bash(rm -rf*)",
      "Bash(git push --force*)",
      "Bash(git reset --hard*)",
      "Bash(npm publish*)",
      "Bash(curl*)",
      "Read(.env*)",
      "Read(**/secrets/**)"
    ],
    "ask": [
      "Bash(git push*)",
      "Bash(npm install*)"
    ]
  }
}
```

Глобы: `*` — любой хвост, `**` — любые пути. Deny всегда приоритетнее allow.

## /permissions в сессии

Интерактивный редактор правил. Изменения записываются в `settings.local.json` — личные, не попадают в git.

## --dangerously-skip-permissions

Флаг отключает все подтверждения. Использовать ТОЛЬКО:

- в свежем Docker-контейнере без доступа к секретам
- в CI на изолированном runner
- на одноразовых VM

НИКОГДА не включай на рабочей машине с доступом к прод и продакшн-ключам.

## Sandbox-рабочий флоу

```bash
docker run --rm -it \
  -v $(pwd):/workspace -w /workspace \
  -e ANTHROPIC_API_KEY \
  --network bridge \
  node:20 bash -c "npm i -g @anthropic-ai/claude-code && claude --dangerously-skip-permissions"
```

Агент внутри контейнера не имеет доступа к хосту. Можно идти на риск.

## Секреты

- Держи `.env` в `deny: Read`
- Используй `direnv` или 1Password CLI для инъекции переменных
- Не храни ключи API в истории сессий (они ложатся в `~/.claude/projects/`)

## Prompt injection

Агент читает всё: README, issues, комментарии в коде, ответы MCP-серверов. Чужой вредоносный текст может попытаться заставить его выполнить опасные команды.

Защита:

- Deny-лист для разрушительных команд
- Ask для push, публикации, установки пакетов
- WebFetch только на доверенные домены
- Не запускай агента на коде из неизвестных источников без изоляции

## Практика

1. Создай `.claude/settings.json` с allow-блоком для Read/Edit/Write и безопасных bash-команд.
2. Добавь deny на `rm -rf`, `git push --force`, `npm publish`.
3. В `settings.local.json` разреши `Bash(git push origin*)` без ask, если уверен.
4. Попробуй в сессии попросить агента прочитать `.env` — убедись, что блокирует.
5. Запусти `claude` в Docker с `--dangerously-skip-permissions` и дай задачу на рефакторинг.
6. Напиши wrapper-скрипт `yolo.sh`, который запускает агента в контейнере.
7. Добавь WebFetch allow-лист с доменами docs.*.
8. Разберись, как `/permissions` показывает эффективный набор правил.

## Чек-лист

- [ ] Проектный `settings.json` в git
- [ ] Deny на разрушительные команды
- [ ] `.env` и секреты в deny
- [ ] Настроен sandbox-запуск для рискованных задач
- [ ] Понимаешь риск prompt injection

---

← [03. Команды](03-commands-modes.md) | [05. Навигация по кодовой базе →](05-codebase-navigation.md)
