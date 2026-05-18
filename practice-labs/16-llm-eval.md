# 🧮 Лаба 16: Evaluation harness для LLM 🔴

## Цель

Построить воспроизводимый пайплайн оценки LLM на доменных задачах: golden set + LLM-as-judge + регрессии. Это фундамент для любых изменений промпта или модели в проде.

## Компоненты

- Golden set: 100–300 примеров с эталонными ответами.
- Промпт судьи с rubric.
- Metrics dashboard.
- Regression suite: на каждый PR — обязательный ран.

## Минимальный пайплайн

1. Собрать golden set: input + reference + (опц) критерии.
2. Реализовать LLM-judge с chain-of-thought rubric.
3. Параллельный запуск эвала (asyncio).
4. Сохранение result.jsonl + diff vs previous run.
5. CI hook: падает PR, если score упал >2%.

## Метрики

- Accuracy/exact match (где возможно).
- LLM-judge score 1–5 с обоснованием.
- Faithfulness / relevance / completeness.
- Win-rate vs baseline (pairwise).
- Cost и latency по каждому примеру.
- Inter-rater agreement с человеческой разметкой (calibrate judge).

## Расширения

- Adversarial set: prompt injection, jailbreak attempts.
- A/B-тесты промптов в бою.
- Ragas / DeepEval / Promptfoo интеграция.
- Slice-based metrics (по языкам, типам вопросов).
- Cost-aware выбор модели (Pareto front).

## Критерии приёмки

- [ ] Golden set ≥1 версия в git.
- [ ] Judge prompt в отдельном файле (версионируется).
- [ ] CI ран на PR.
- [ ] Dashboard с историей метрик.
- [ ] Inter-rater agreement измерен (≥0.7).
- [ ] Cost per eval известен.

## Анти-паттерны

- ❌ Judge на той же модели, что и тестируемая (bias).
- ❌ Голден сет из LLM без ручной проверки.
- ❌ Promt judge без rubric — судебные решения «плавают».
- ❌ Нет версии (нельзя откатиться).
- ❌ Игнор latency/cost («лучше» на 5%, в 10х дороже).

---

[← Назад к Practice Labs](./README.md)
