# 🏗️ ML System Design: 10 кейсов с разбором

Самый сложный этап ML-собеседования: дают задачу, 60 минут на проектирование. Здесь — типовые кейсы 2026 года с разбором каждого по единой схеме.

## 🎯 Универсальная схема ответа

Используйте её для **любого** ML system design вопроса:

1. **Уточнить требования (5 мин).** Бизнес-цель, метрики успеха, constraints (latency, throughput, budget, regulatory).
2. **Прикинуть масштаб (3 мин).** Сколько пользователей, запросов, данных. На чём упрёмся.
3. **Выбрать ML-формулировку (5 мин).** Классификация / регрессия / ranking / generation. Что предсказываем, что метрика.
4. **Данные (10 мин).** Источники, разметка, frequency, edge cases.
5. **Модель (10 мин).** Baseline → улучшение. Train pipeline.
6. **Eval (5 мин).** Offline (метрики) + online (A/B).
7. **System (10 мин).** Архитектура: feature store, training, serving, monitoring.
8. **Iteration (5 мин).** Что мониторим, когда retrain, как catch drift.

**Правило:** не молчите. Каждый шаг — обсуждайте вслух с интервьюером. Спрашивайте уточнения.

---

## Кейс 1: Рекомендации YouTube/Netflix

**Постановка:** «Спроектируйте систему рекомендаций видео для главной страницы».

**Уточняющие вопросы:**
- Главная или related videos? — главная.
- Сколько слотов? — топ-20.
- Cold start: новые пользователи / новые видео?
- Метрика успеха: watch time / clicks / retention?
- Latency бюджет: 100ms p95.

**Масштаб:**
- 100M DAU × 10 запросов = 1B запросов/день = ~12K RPS.
- Каталог: 100M видео.

**ML-формулировка:** двухэтапная **retrieval + ranking**.
- Retrieval: из 100M → 1000 кандидатов.
- Ranking: из 1000 → 20 + сортировка.

**Данные:**
- Implicit: clicks, watch time, skips, likes.
- Контент: title, description, tags, embeddings видео (CLIP-style).
- User: история, demographics, time of day.

**Модели:**

*Retrieval (Two-Tower):*
- User tower: user features → 128-dim vector.
- Item tower: item features → 128-dim vector.
- Train на in-batch negatives (contrastive).
- Index: FAISS HNSW для ANN search.

*Ranking:*
- Gradient boosting (LightGBM) или DLRM-style нейросеть.
- Фичи: user×item interactions, дата, время, контекст.
- Multi-task: predict click + watch time + completion.
- Output: weighted sum под бизнес-метрику.

**Eval:**
- Offline: NDCG@20, Recall@1000 для retrieval.
- Online: A/B на watch time, retention, daily sessions.
- Beyond-accuracy: diversity (intra-list similarity), novelty, freshness.

**System:**
- Feature store (Feast) для user/item features.
- Embeddings обновляем daily (batch).
- Ranking model: online inference, batch предсказание для топ-кандидатов.
- Cache top-N для inactive users.

**Monitoring:**
- Drift в фичах (PSI).
- Watch time per session по сегментам.
- Coverage каталога (% видео получают показы).
- Cold start metrics.

**Iteration:**
- Retrain retrieval weekly, ranking daily.
- A/B новых моделей в shadow.
- Multi-armed bandit для exploration.

---

## Кейс 2: Поиск Instagram/TikTok

**Постановка:** «Спроектируйте поиск контента по запросу пользователя».

**Уточнения:**
- Что ищем: посты, авторов, хэштеги, видео? — всё.
- Multi-modal (текст + картинка)? — да.
- Personalization? — да, по истории.
- Latency: 200ms p95.

**Масштаб:**
- 1B пользователей, 100K RPS на поиск.
- 10B постов в индексе.

**Архитектура двухэтапная:**

*Retrieval (sub-100ms):*
- **Inverted index** (Lucene/Elasticsearch) для keyword match.
- **Vector search** через multi-modal embeddings (CLIP-like).
- **Hybrid:** RRF слияние.
- Кандидатов: 1000-10000.

*Ranking (50-100ms):*
- LightGBM / нейросеть с фичами:
  - Match score (BM25, cosine sim).
  - Quality signals (engagement, recency, account quality).
  - Personalization (user-author affinity, прошлые взаимодействия).
- Multi-objective: relevance + diversity + freshness.

**Edge cases:**
- Misspellings → query rewriting (LLM или char-level model).
- Synonyms → query expansion.
- Trending queries → отдельный hot cache.
- Long-tail queries → embeddings справляются лучше, чем keyword.

**Eval:**
- Offline: NDCG, MRR на judged queries.
- Online: CTR, dwell time, abandonment rate.
- Human eval на repr-sample queries раз в неделю.

**System components:**
- Query understanding service (LLM + rules).
- Retrieval service (ES + vector DB).
- Ranker service.
- Feature service (real-time + batch).
- Logs → training data pipeline.

---

## Кейс 3: Fraud detection для платёжной системы

**Постановка:** «Detect fraudulent transactions в реальном времени».

**Уточнения:**
- Какие виды fraud? — карточный, account takeover, money laundering.
- Latency: < 100ms.
- Cost of FP / FN: FN — потеря денег, FP — раздражение клиента + поддержка.
- Регуляторика: PCI DSS, explainability.

**Масштаб:**
- 10K транзакций/сек peak.
- Fraud rate ~0.1% (огромный дисбаланс).

**ML-формулировка:** binary classification + risk score.

**Данные:**
- Транзакция: amount, merchant, time, location, channel.
- User: history, age account, прошлые fraud.
- Merchant: risk score, частота fraud.
- Device: IP, browser fingerprint.
- Graph features: связи between accounts.

**Модели:**
- **Real-time:** LightGBM/XGBoost на 100+ фичах. Inference < 50ms.
- **Async:** более глубокая модель (нейросеть, GNN) для дополнительной проверки.
- **Rules engine:** жёсткие правила для известных паттернов (KYC, sanctions).

**Метрики:**
- **Recall** при precision = 0.95 (бизнес: блокировать только если уверены).
- Money saved / blocked.
- False positive rate (UX impact).

**Class imbalance:**
- Stratified sampling.
- class_weight / focal loss.
- Не SMOTE — на time-series это плохо работает.

**System:**
- Online feature store (Redis) с realtime aggregations (last hour spending, velocity).
- Model serving: low-latency gRPC service.
- Decision pipeline: ML score + rules → action (allow / step-up auth / block).
- Audit log (regulatory).

**Monitoring:**
- Score distribution drift (PSI).
- Fraud rate (catch rate vs feedback).
- Latency p99.
- Bias по сегментам (regulatory).

**Iteration:**
- Retrain weekly.
- Fast-track для новых fraud patterns (active learning).
- Adversarial: фродстеры адаптируются.

---

## Кейс 4: LLM-чатбот для поддержки

**Постановка:** «Build customer support chatbot для e-commerce».

**Уточнения:**
- Языки: русский + английский.
- Объём: 100K запросов/день.
- Задачи: статус заказа, возвраты, FAQ, эскалация на оператора.
- Constraints: privacy (PII), accuracy критична для денежных операций.

**Архитектура:**

```
User → Router → {
    Simple FAQ → RAG over docs
    Order info → Function calling (DB queries)
    Refund → LLM + human approval
    Complex → Escalate to human
}
```

**Компоненты:**

*Router:* классификатор намерения (intent classification). Маленький fine-tuned BERT или LLM в zero-shot.

*RAG:* индекс над FAQ и документацией.
- Chunking: по разделам docs.
- Embeddings: multilingual (bge-m3).
- Hybrid search + reranker.

*Function calling:* для structured запросов (order status, account info).
- Tools: get_order, cancel_order (с approval), get_returns_policy.
- Validation arguments.

*LLM:* Claude/GPT для генерации ответов. Fine-tune на корпусе исторических ответов поддержки для tone.

**Safety:**
- PII detection на входе/выходе.
- Guardrails на «гарантирую возврат», advice по health, юридические вопросы.
- Human escalation для high-stakes.

**Eval:**
- Offline: golden dataset с эталонными ответами. LLM-as-judge.
- Online: CSAT, resolution rate, escalation rate, time to resolution.
- Side-by-side comparison с человеческим оператором.

**System:**
- Async architecture (websocket для streaming).
- Conversation state в Redis.
- Logs → Langfuse.
- Feedback loop: thumbs up/down → данные для дообучения.

**Cost:**
- 100K msg/day × 1500 tokens avg = 150M tokens/day.
- На GPT-4o: ~$1500/day. На Claude Haiku + GPT-4o router: ~$300/day.

---

## Кейс 5: News feed ranking (Facebook/LinkedIn)

**Постановка:** «Спроектируйте ranking новостной ленты».

**Уточнения:**
- Цели: meaningful interactions, time spent, retention.
- Anti-goals: rage clicks, misinformation, low-quality content.
- Латентность: ~200ms для рендера feed.

**Архитектура:**

1. **Inventory:** из миллиардов постов отобрать кандидатов от друзей/follows за последние N дней (~5K кандидатов).
2. **First-pass ranker:** лёгкая модель (LightGBM), top-500.
3. **Heavy ranker:** глубокая модель (MMoE — Multi-gate Mixture of Experts), top-100.
4. **Re-ranker:** diversity, freshness, integrity constraints, top-25.

**Multi-objective learning:**
- Heads: P(click), P(like), P(comment), P(share), P(hide), P(report).
- Final score: Σ weight_i · prediction_i. Weights — бизнес-баланс.

**Integrity (anti-fake-news):**
- Classifier для clickbait, low-quality, потенциальная misinfo.
- Down-rank или фильтр.

**Feedback loops:**
- Если weight click высокий → больше clickbait → токсичный feedback loop.
- Решение: balance heads, явные negative signals (hide), human review.

**Eval:**
- Offline: NDCG per objective.
- Online: time spent, sessions/week, hide rate, report rate.
- Long-term: 6-week retention, user surveys.

**System:**
- Feature store с realtime aggregations.
- Embeddings для users, items, content.
- Model serving с auto-scaling.
- Logging: каждое impression + action.

---

## Кейс 6: Image classification at scale (модерация контента)

**Постановка:** «Detect harmful content (nudity, violence, spam) на UGC платформе».

**Уточнения:**
- Объём: 100M фото/день.
- Categories: nudity, violence, hate symbols, spam, child safety (priority).
- SLA: < 5 sec для review, < 100ms для критичных категорий.

**Pipeline:**

```
Upload → Hashing check (PhotoDNA для CSAM) → 
ML model (multi-label) → 
{score < low: auto-allow,
 low < score < high: human review queue,
 score > high: auto-block + appeal flow}
```

**Модели:**
- **CSAM:** hashing (PhotoDNA), zero tolerance.
- **General:** multi-label CNN/ViT. EfficientNet-B3 или ViT-base.
- Fine-tuned на собственных данных платформы.
- Per-category thresholds.

**Train data:**
- Labeled by trust & safety team (sensitive work).
- Active learning на uncertain samples.
- Adversarial augmentation: faces, text overlays.

**Eval:**
- Per-class precision/recall.
- Cost-sensitive: FN на CSAM недопустим, на spam — приемлем.
- Bias по demographics (skin tone, gender).
- Adversarial robustness.

**System:**
- Batch inference (Triton).
- GPU pool с autoscaling.
- Human review tool (annotation UI).
- Appeal pipeline.
- Audit log (legal).

**Iteration:**
- Weekly retrain на новых labeled data.
- Active learning: показывайте человеку border cases.
- Adversarial monitoring: новые попытки обхода.

---

## Кейс 7: Real-time speech-to-text для звонков

**Постановка:** «Real-time transcription для colls support».

**Уточнения:**
- Языки: русский + английский.
- Latency: < 500ms для каждой фразы (streaming).
- Quality: WER < 10% для clear audio.
- Privacy: on-prem option для enterprise.

**Архитектура:**

*Audio pipeline:*
- VAD (voice activity detection): отделяем речь от тишины.
- Diarization: кто говорит (operator vs client).
- Streaming ASR (Whisper-streaming, NVIDIA Parakeet).

*Post-processing:*
- Punctuation restoration.
- Inverse text normalization (числа, даты).
- Domain adaptation: словарь компании.

**Модели:**
- **ASR:** Whisper-large-v3 fine-tuned на доменных данных. Или Parakeet (быстрее).
- **Diarization:** pyannote.audio.
- **Streaming:** chunk-based inference с overlap.

**Eval:**
- WER overall и per speaker.
- Latency p95 на чанк.
- Domain-specific: product names, account numbers.

**System:**
- WebRTC для audio streaming.
- ASR service на GPU (Triton).
- Real-time results via WebSocket.
- Storage: full transcripts + audio (с retention policy).

**Privacy:**
- PII detection в transcript (имена, карты, телефоны).
- Redaction в storage.
- Encryption at rest и in transit.

---

## Кейс 8: Search autocomplete

**Постановка:** «Спроектируйте search suggestions like Google».

**Уточнения:**
- Latency: < 50ms (suggestion appears as user types).
- Personalization: да (история).
- Languages: many.
- Spam-free: нет токсичности, NSFW.

**Подход:**

*Storage:* Trie / FST (Finite State Transducer) для prefix lookup.

*Ranking signals:*
- Query frequency (count).
- Recency (time decay).
- Personal: user history.
- Geographic: trending in region.
- Contextual: previous queries в сессии.

**Pipeline:**
- Logs queries → aggregation → trie update (daily).
- User typing → trie lookup (top-100 prefix matches) → ranker → top-10.

**Spam protection:**
- Block lists.
- ML classifier для toxic/NSFW.
- Min frequency threshold.

**ML использование:**
- Embedding-based для semantic suggestions (не prefix).
- Personalization model (LightGBM на user × suggestion фичах).
- Spell correction (BERT-based для context-aware).

**Eval:**
- CTR на suggestions.
- MRR (на каком ranke выбрали).
- Latency p99.
- Coverage (% запросов получают suggestions).

**Edge cases:**
- Cold start: новый пользователь → popular suggestions.
- New trending event → fast pipeline для свежих queries (hourly update).
- Multilingual: language detection + per-language trie.

---

## Кейс 9: Personalized notifications

**Постановка:** «Когда и что присылать в push для retention».

**Уточнения:**
- Channels: push, email, in-app.
- Goal: retention без annoying.
- Frequency cap: max 3/day per channel.

**ML задачи:**

1. **Send time prediction:** когда юзер активен.
   - Per-user time histogram.
   - ML на signals: timezone, app open patterns, day of week.
   - Output: ranked time slots.

2. **Content selection:** какое уведомление прислать.
   - Candidates: новые посты, рекомендации, social events, promotions.
   - Ranker: predict P(open notification | user, content, time).

3. **Frequency optimization:**
   - Uplift modeling: «если пришлю — откроет ли? Vs если не пришлю».
   - Long-term: не лучший CTR, а unsubscribe rate, retention.

**Pipeline:**

```
Trigger (event / scheduled) → 
Eligible users → 
Send-time check (within active hours, not over cap) → 
Content selection → 
Personalization ranker → 
A/B framework → 
Send (FCM/APNS) → 
Track open/dismiss/unsubscribe
```

**Eval:**
- Open rate per notification type.
- Long-term: weekly active users, churn rate.
- Unsubscribe rate (critical).
- Uplift: incremental sessions vs control group.

**Anti-patterns:**
- Оптимизировать только CTR → spam.
- Игнорировать negative signals (unsubscribe, mute).
- Не учитывать context (user в meeting).

---

## Кейс 10: On-device speech recognition (mobile)

**Постановка:** «Спроектируйте on-device wake word + speech recognition».

**Уточнения:**
- Constraints: telephone, no internet, battery.
- Latency: < 100ms для wake word detection.
- Wake word accuracy: 99%+ true positive, < 1 false trigger/day.
- ASR quality: WER < 15% на коротких командах.

**Архитектура:**

*Always-on stage:*
- **Wake word detector:** маленькая CNN (~50K params), всегда работает.
- Power-efficient (DSP chip).
- Output: triggered or not.

*Triggered stage:*
- **Streaming ASR:** компактная модель (RNN-T или Conformer), ~10M params.
- On-device inference.
- Optional fallback на cloud для сложных запросов.

**Модели:**
- **Wake word:** keyword spotting (DSCNN).
- **ASR:** RNN-T (RNN Transducer) — стандарт для on-device.
- Quantized to int8.
- Через TFLite / CoreML / ONNX Runtime.

**Train:**
- Wake word: 100K+ recordings (positive) + larger negative dataset (любой звук).
- ASR: огромный корпус, fine-tune на on-device domain (короткие команды).
- Adversarial: noise robustness (TV in background, music).

**Eval:**
- True positive rate (wake word).
- False alarm per hour.
- Latency on target device.
- WER on commands.
- Power consumption (mAh per hour).

**System:**
- OTA model updates через app store.
- Telemetry (privacy-preserving aggregations).
- Fallback: cloud ASR для complex queries (с user permission).

---

## 🎯 Финальные советы по ML system design

1. **Не молчите.** Думайте вслух. Спрашивайте уточнения. Интервьюер хочет видеть процесс.

2. **Начинайте с простого.** Бейзлайн → улучшения. «Сначала LightGBM, потом добавим нейросеть, если нужно».

3. **Обсуждайте trade-offs.** «Этот подход быстрее, но требует больше памяти. На нашем масштабе — приемлемо.»

4. **Не забывайте про данные.** 80% реальной работы — данные, разметка, фичи. Поверхностный ответ — провал.

5. **Думайте про eval.** Offline + online. Метрики бизнеса, не только модели.

6. **Затрагивайте non-ML части.** Caching, sharding, fault tolerance. Это показывает зрелость.

7. **Признавайте ограничения.** «Этот подход не сработает, если...». Это сильнее, чем «всё всегда работает».

8. **Используйте опыт.** «Я делал похожее на проекте X, там...». Резко повышает доверие.

---

## 📚 Ресурсы для углублённой подготовки

- **Книга:** *Machine Learning System Design Interview* — Ali Aminian. Структурированные разборы.
- **Книга:** *Designing Machine Learning Systems* — Chip Huyen. Фундамент.
- **Канал:** [Recsys Channel](https://www.youtube.com/@RecSysChannel) — глубокие разборы.
- **Блоги компаний:** Meta AI, Google AI, Pinterest Engineering, Airbnb Engineering, Spotify Research.
- **Доклады:** RecSys, KDD, MLSys конференции.

> 🎯 ML system design — самый сложный этап. Подготовьте 10 кейсов глубоко, и вы пройдёте любой собес. Главное — структура ответа и trade-offs.
