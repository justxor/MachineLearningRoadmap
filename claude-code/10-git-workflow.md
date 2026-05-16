# 10. Git, ветки, PR

## Агент в git

Claude Code хорошо владеет git CLI. Он способен:

- Писать осмысленные коммиты по диффу
- Разбивать большой дифф на логические коммиты
- Делать interactive rebase
- Читать blame и git log
- Открывать PR через `gh`

## Conventional Commits

Стандартный формат:

```
feat(auth): add email verification on signup
fix(payments): handle Stripe rate limit with backoff
refactor(db): extract migration runner
docs: update onboarding
test(users): add edge cases for unicode emails
chore(deps): bump fastify to 4.27
```

Пропиши в CLAUDE.md «use Conventional Commits» — агент будет следовать.

## Правильный commit-промпт

```
> Сделай `git status` и `git diff`.
> Разбей изменения на логические коммиты. Предложи план. Ничего не коммить без моего ok.
```

Агент покажет план вида:

```
1. feat(api): add /users/:id/verify endpoint
   - src/routes/users.ts (lines 45-78)
   - src/services/verification.ts (new)
2. test(api): cover verify endpoint
   - tests/users.verify.test.ts (new)
3. docs(api): document verification flow
   - docs/api/verification.md (new)
```

## Interactive staging

Claude умеет `git add -p`-стиль:

```
> Из текущего diff выбери только изменения в src/services/, сделай коммит.
> Потом отдельный коммит для тестов.
```

## PR через gh

```
> Пушни ветку и открой PR через gh pr create.
> Title: feat: email verification
> Body включает: Summary, Changes, How to test, Risks.
> Не сливай, не merge.
```

Шаблон PR (положи в `.github/pull_request_template.md`):

```markdown
## Summary
<1–2 предложения>

## Changes
- что изменилось

## How to test
- команды и сценарии

## Risks
- что может сломаться
```

## Worktrees для параллельных задач

Две задачи сразу в одном репо:

```bash
git worktree add ../project-fix-auth fix/auth
git worktree add ../project-feat-search feat/search
```

Запусти два `claude` в разных окнах терминала. Работают параллельно без конфликта веток.

## Код-ревью PR

```
> Посмотри PR #123 в этом репо (через `gh pr diff 123` и `gh pr view 123`).
> Сделай ревью: баги, стиль, вопросы. Не комментируй, просто напиши отчёт.
```

## Что не давать делать

- Force push в чужие ветки
- Push в main/master напрямую (заблокируй в settings.json)
- `git reset --hard` без подтверждения
- Мерджить без ревью (`gh pr merge` — в ask-лист)

## Практика

1. Сделай большой дифф, попроси разбить на 5–7 коммитов.
2. Настрой commitlint и добавь husky pre-commit хук.
3. Попроси написать PR description по диффу.
4. Открой PR через `gh pr create` из сессии.
5. Создай worktree и запусти два агента на разных ветках.
6. Попроси сделать interactive rebase из 5 коммитов в 1 squashed.
7. Напиши review чужого PR от имени агента и сравни со своим.
8. Добавь `.github/pull_request_template.md` и проверь, что gh его подхватывает.

## Чек-лист

- [ ] Conventional Commits в CLAUDE.md
- [ ] Агент разбивает дифф на логические коммиты
- [ ] PR открываешь через `gh`
- [ ] Шаблон PR в репо
- [ ] Worktree для параллельных задач пробовал
- [ ] Force push запрещён

---

← [09. Рефакторинг](09-refactoring.md) | [11. MCP: Model Context Protocol →](11-mcp-basics.md)
