# 03. SQL для Data Science

SQL — рабочий язык 80% времени аналитика и почти любого DS в продукте. Не выучив оконные функции и CTE, вы будете выводить всё в pandas и ждать часами.

## Что уметь

- SELECT, WHERE, GROUP BY, HAVING.
- JOIN: INNER, LEFT, FULL, CROSS, ANTI-JOIN через LEFT JOIN + IS NULL.
- Оконные функции: ROW_NUMBER, RANK, LAG, LEAD, SUM() OVER (...).
- CTE и рекурсивные CTE.
- Подзапросы vs JOIN.
- Даты: DATE_TRUNC, INTERVAL, EXTRACT.
- NULL-логика и COALESCE.

## Шаблоны

**Ретеншн по когортам**

```sql
WITH first_visit AS (
  SELECT user_id, MIN(DATE_TRUNC('week', ts)) AS cohort
  FROM events GROUP BY user_id
)
SELECT cohort,
       DATE_TRUNC('week', e.ts) AS week,
       COUNT(DISTINCT e.user_id) AS active_users
FROM events e JOIN first_visit f USING (user_id)
GROUP BY 1, 2 ORDER BY 1, 2;
```

**Дедупликация по последней записи**

```sql
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC) AS rn
  FROM user_events
) t WHERE rn = 1;
```

## Оптимизация

- Не SELECT *.
- Индексы: понимать принцип, читать EXPLAIN ANALYZE.
- LIMIT при разведке.
- Агрегируйте раньше, чем джойните большие таблицы.

## Антипаттерны

- Тянуть всё в pandas и там фильтровать.
- Копировать 5 вложенных подзапросов вместо CTE.
- DISTINCT, чтобы «починить» плохой JOIN.

## Практика

1. Решите 30 задач на SQLZoo / LeetCode SQL (medium+).
2. Напишите запрос DAU/WAU/MAU для вымышленной таблицы events.
3. Сделайте воронку из событий (signup → first_action → purchase).
4. Посчитайте среднее время между событиями через LAG.
5. Исследуйте EXPLAIN ANALYZE любого своего тяжёлого запроса.
6. Соберите витрину (mart) из 3 таблиц с комментариями.

## Чек-лист

- [ ] Пишу оконные функции без Гугла.
- [ ] Читаю план запроса.
- [ ] Различаю INNER/LEFT/FULL JOIN и их эффект на кардинальность.

[← 02](02-tools.md) • [→ 04. Pandas & NumPy](04-pandas-numpy.md)
