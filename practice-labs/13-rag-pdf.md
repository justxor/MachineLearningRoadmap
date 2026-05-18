# 📚 Лаба 13: RAG over PDF с цитированием источников 🟡

## Цель

Построить RAG-систему по PDF-документам, которая отвечает на вопросы со ссылкой на конкретные страницы. Понять, как работает retrieval, chunking, reranking.

## Данные

- 5–20 PDF-документов (отчёты, доки, книги).
- Golden set: 30–50 вопрос-ответ пар со ссылками на источник.

## Минимальный пайплайн

1. Парсинг PDF: pypdfium2 / unstructured (сохраняя страницы).
2. Chunking: 500–1000 токенов с overlap 100.
3. Embeddings: bge-small-en или e5-small (русский: ru-en-RoSBERTa).
4. Индекс: FAISS или Qdrant.
5. Retriever top-k=10 + reranker (bge-reranker) → top-3.
6. LLM (GPT-4o-mini или local): генерация ответа с cite-разметкой.
7. Streamlit/Gradio UI.

## Код: chunk + embed + index

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = []
for page_num, page_text in pages:
    for c in splitter.split_text(page_text):
        chunks.append({"text": c, "page": page_num})

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
```

## Метрики

- Recall@k (нужный chunk в топ-k).
- MRR (Mean Reciprocal Rank).
- Faithfulness (LLM-as-judge): ответ основан на источнике?
- Answer relevance, context precision, context recall (ragas).
- Latency p50/p95.

## Расширения

- Hybrid retrieval: BM25 + dense + reciprocal rank fusion.
- Multi-query: LLM разлагает вопрос на подвопросы.
- HyDE (Hypothetical Document Embeddings).
- Иерархический chunking (summary + details).
- Citation post-processing: выводить «страница 12, параграф 3».

## Критерии приёмки

- [ ] Recall@5 >0.8 на golden set.
- [ ] Faithfulness >0.9 (LLM-as-judge или ragas).
- [ ] Каждый ответ включает цитату с указанием страницы.
- [ ] UI работает, latency p95 <5с.
- [ ] Eval-скрипт в репо (выводит все метрики).

## Анти-паттерны

- ❌ Демо без измерения качества («кажется ок»).
- ❌ Чанки по страницам без overlap.
- ❌ LLM «галлюцинирует» без режима «не знаю».
- ❌ Нет reranker — выдаёт нерелевантные chunks.

---

[← Назад к Practice Labs](./README.md)
