# Капстон 3. Production-ready RAG-ассистент

> Делается **после уроков 13-18**. Цель: построить ассистента, который отвечает на вопросы по вашим документам. Это самый востребованный паттерн LLM-приложений в индустрии в 2025.

## Что должно быть на выходе

- 🗂️ GitHub-репозиторий, например `my-rag-assistant`.
- 📚 Корпус из ваших документов (PDF / Markdown / HTML).
- 🔍 Векторная БД с embeddings.
- 🤖 LLM-инференс (через API или локально через Ollama/vLLM).
- 🚀 FastAPI с эндпоинтами `/chat` и `/health`.
- 🐳 Docker-образ < 1 ГБ.
- 📊 Базовый мониторинг latency и количества обращений.

## Выбор предметной области

Возьмите то, по чему **можно проверить ответы**:

- **Документация одного фреймворка** — например, FastAPI / Pandas / PyTorch.
- **Корпус нормативки или регламентов** — ГОСТ, ПДД, корпоративные политики.
- **Wiki вашей компании / коллекция meeting notes**.
- **Книги по узкой теме** (например, всё, что есть про CRISPR).
- **Корпус ваших Notion / Obsidian-заметок** — личный ассистент.

Источников должно быть на 1-50 MB чистого текста. На 1 GB корпуса будет работать так же, просто индекс дольше строится.

## Архитектура

```
[ Пользователь ]
       ↓ /chat
[  FastAPI  ]
       ↓
[ Reranker (опционально) ]
       ↓
[ Retriever: emb-модель + векторная БД ]
       ↓ top-k чанков
[       LLM с RAG-промптом       ]
       ↓
[ Ответ + источники + латенси ]
```

## Стек

- **sentence-transformers** или **`intfloat/multilingual-e5-large`** — embeddings.
- **Qdrant** или **ChromaDB** или **FAISS** — векторная БД.
- **LLM:** либо API (OpenAI, Anthropic, OpenRouter), либо локально `Mistral-7B` через Ollama / vLLM.
- **FastAPI + Uvicorn** — сервис.
- **Docker** — упаковка.
- **`langchain`** опционально (но я бы не брал — много магии).
- **Prometheus + Grafana** (опционально, для production-уровня).

## Структура репозитория

```
my-rag-assistant/
├── README.md
├── docker-compose.yml         # FastAPI + Qdrant в одном compose
├── Dockerfile
├── pyproject.toml
├── src/
│   ├── ingest.py              # парсинг документов, chunking, индексация
│   ├── retriever.py           # поиск top-k через embeddings + reranker
│   ├── llm.py                 # обёртка над LLM (API или Ollama)
│   ├── prompt.py              # RAG-промпт-шаблоны
│   └── api.py                 # FastAPI app
├── tests/
│   └── test_e2e.py            # на 5-10 «известных» вопросах
└── eval/
    ├── questions.jsonl        # 30-50 ваших вопросов с эталонными ответами
    └── run_eval.py
```

## Пошаговый план (10-14 дней)

**День 1-2. Ingest pipeline.**

- Парсинг PDF (`pdfplumber`, `unstructured`), Markdown, HTML.
- Чанкинг: 500-1000 токенов с overlap 50-100. **Это самое важное**, что влияет на качество.
- Сохраняйте `source` (файл + страница / якорь) — потом понадобится для цитат.

**День 3. Embeddings и БД.**

- Возьмите `intfloat/multilingual-e5-large` (хорошо работает на русском).
- Прокачайте все чанки → векторы → Qdrant с metadata (`source`).
- Скрипт ingest должен быть **идемпотентным**: можно перезапустить.

**День 4. Retriever.**

- Функция `retrieve(query, k=5)` → top-k чанков.
- Embedding query то же, что у документов.
- Замерьте `recall@5` на 20 размеченных вопросах вручную.

**День 5. Reranker (опционально, но сильно поднимает качество).**

- `BAAI/bge-reranker-v2-m3` — cross-encoder.
- Retrieve k=20 → rerank до top-5.
- Сравните метрики до/после.

**День 6-7. LLM-обёртка.**

- Унифицированный интерфейс `chat(messages, max_tokens, temperature) -> str`.
- Поддержка нескольких бэкендов: OpenAI API, локальный Ollama.
- Streaming (FastAPI `StreamingResponse`) — пользователю удобнее.

**День 8. RAG-prompt.**

```
Ты — ассистент по [тема]. Отвечай ТОЛЬКО на основе фрагментов из источников.
Если ответа нет — скажи «не знаю».

ИСТОЧНИКИ:
[1] {chunk_1}
[2] {chunk_2}
...

ВОПРОС: {query}

ОТВЕТ (с обязательными ссылками вида [1], [2]):
```

- Запретите выдумывать ссылки.
- Включите в ответ `source`-секцию с цитатами.

**День 9. FastAPI и Docker.**

- POST `/chat` принимает `{"query": str, "history": [...]}`, возвращает `{"answer": str, "sources": [...]}`.
- GET `/health` для k8s readiness.
- Логирование запросов/ответов в файл (без PII).
- Multi-stage Dockerfile с poetry / uv.
- docker-compose: ваш сервис + Qdrant.

**День 10-11. Eval.**

- Соберите 30-50 пар `(вопрос, эталонный ответ)`. Это самая важная часть проекта.
- Метрики:
  - `retrieval recall@5` — есть ли релевантный чанк в top-5.
  - **LLM-as-a-judge**: подавайте `(question, gold_answer, predicted_answer)` в GPT-4 и просите оценить от 1 до 5.
  - **Faithfulness**: процент ответов, где все утверждения подкреплены источниками.
- Зафиксируйте baseline. Каждое улучшение pipeline сверяйте с ним.

**День 12. Мониторинг.**

- Prometheus middleware для FastAPI (`/metrics`).
- Метрики: запросы/сек, p95 latency, доля «не знаю», доля ответов без источников.
- Grafana dashboard со скриншотом в README.

**День 13-14. README и деплой.**

- README на 3-5 экранов с архитектурной схемой и метриками.
- Деплой на бесплатный/дешёвый VPS (Render, Railway, Fly.io) или Hugging Face Spaces.
- Live-demo с 3-5 готовыми примерами вопросов.

## Метрики, обязательные в README

- **Размер корпуса:** документов, токенов, чанков.
- **Latency:** p50, p95, p99 на `/chat`.
- **Recall@5 retriever.**
- **LLM-judge score:** среднее по eval-датасету.
- **Стоимость:** на 1000 запросов (если API), в $/мес.

## Чек-лист готовности к собеседованию

- [ ] Можно дать ссылку на live-демо.
- [ ] Можно дать ссылку на репо, и кто-то другой запустит `docker compose up` без вопросов.
- [ ] Есть честный раздел «Limitations» — где модель галлюцинирует, где промахивается retriever.
- [ ] Есть eval с конкретными цифрами, а не «работает хорошо».
- [ ] Объяснён выбор каждого компонента: почему e5, почему Qdrant, почему такой размер чанка. Это любимый вопрос на собесе.

## Расширенные опции

- **Hybrid search** = BM25 + dense + reranker. На многих задачах это +5-10% recall.
- **Query rewriting:** LLM переписывает вопрос пользователя в 2-3 reformulations, ищем по всем.
- **Memory:** многоходовой диалог — складывайте предыдущие `Q/A` в контекст.
- **Свой fine-tuned reranker** на ваших размеченных парах (если их > 500).
- **Streaming + WebSocket** интерфейс вместо REST.

## Полезные ссылки

- [Qdrant docs](https://qdrant.tech/documentation/).
- [`sentence-transformers`](https://www.sbert.net/).
- [Anthropic «contextual retrieval»](https://www.anthropic.com/news/contextual-retrieval).
- [Ollama](https://ollama.ai/) — локальный LLM-сервер.

---

[← Все капстоны](./README.md) · [README курса](../README.md)
