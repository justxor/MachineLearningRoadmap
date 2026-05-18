# 🔍 Лаба 23: Семантический поиск с reranker’ом 🟡

## Цель

Построить hybrid search (BM25 + dense) с финальным cross-encoder reranker. Научиться балансировать качество и скорость.

## Датасет

- MS MARCO passages.
- BEIR benchmark.
- Свой корпус документов + 100–500 query-doc пар для eval.

## Минимальный пайплайн

1. Indexing: BM25 (Elasticsearch/Tantivy) + dense (bge-small + FAISS).
2. Hybrid retrieval: оба источника → reciprocal rank fusion top-50.
3. Reranker: bge-reranker-base на top-50 → top-10.
4. Eval: nDCG@10, Recall@100, MRR.
5. Latency по стадиям.

## Код: RRF

```python
def rrf(rankings_list, k=60):
    scores = {}
    for rankings in rankings_list:
        for rank, doc_id in enumerate(rankings):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## Метрики

- nDCG@10 (главная для ranking).
- Recall@100 (ретривер хорош сверху).
- MRR.
- Latency p50/p95 по стадиям (BM25, dense, RRF, rerank).
- Cost per query.

## Расширения

- Query expansion через LLM.
- ColBERT (late interaction).
- Domain-specific fine-tuning encoder.
- Quantization dense эмбеддингов (int8).
- Caching popular queries.
- Feedback loop: логировать clicks → дообучать reranker.

## Критерии приёмки

- [ ] BM25 baseline + hybrid + reranker — 3 варианта в таблице.
- [ ] Hybrid побеждает BM25 и dense по отдельности.
- [ ] Reranker улучшает nDCG на ≥3pp.
- [ ] Latency p95 <300мс.
- [ ] UI или API работает.
- [ ] Eval воспроизводим (скрипт в репо).

## Анти-паттерны

- ❌ Оценка без baseline («работает хорошо» — по сравнению с чем?).
- ❌ Reranker на top-1000 (слишком медленно).
- ❌ Неодинаковый normalization у BM25 и dense scores при фузии.
- ❌ Игнор stop words и языковых особенностей (русский с morph).

---

[← Назад к Practice Labs](./README.md)
