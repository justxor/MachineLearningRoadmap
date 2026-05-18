# Капстон 3. Multi-modal RAG над изображениями и документами

Объединяем CV + LLM в production-ready приложение: поиск по фотобанку / документам / каталогу товаров через текст или картинку, с answer generation через VLM.

## Идеи задач

- **Поиск по корпоративному фотобанку** (визуальный + текстовый запрос).
- **E-commerce visual search** (картинка товара → похожие в каталоге).
- **Knowledge base assistant** для технической документации с диаграммами и схемами.
- **Medical image search** для radiology cases.
- **Real estate** assistant: найти квартиры по описанию + фото.
- **Educational:** «найди диаграммы про energy efficiency в учебниках».

## Архитектура

```
┌─────────────────────────────────────────────┐
│   Ingestion (offline / batch)               │
├─────────────────────────────────────────────┤
│  Documents/Images →                          │
│  ├─ Text extraction (PaddleOCR / Tesseract) │
│  ├─ Image embeddings (CLIP/SigLIP)          │
│  ├─ Text embeddings (bge-m3)                │
│  └─ Metadata extraction (VLM)               │
│      ↓                                       │
│  [Vector DB: Qdrant]                         │
│      ↓                                       │
│  [Metadata DB: PostgreSQL]                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│   Query (online)                            │
├─────────────────────────────────────────────┤
│  User query (text / image / both) →         │
│  ├─ Encoder → query embeddings              │
│  ├─ Vector search → top-50 candidates       │
│  ├─ Reranking (cross-encoder)               │
│  ├─ Top-5 → VLM (Qwen2-VL / GPT-4o)         │
│  └─ Answer with citations                    │
└─────────────────────────────────────────────┘
```

## Структура проекта

```
multimodal-rag/
├── README.md
├── docker-compose.yml
├── ingest/
│   ├── pipeline.py
│   ├── ocr_module.py
│   ├── embedding_module.py
│   └── metadata_extractor.py
├── search/
│   ├── retriever.py
│   ├── reranker.py
│   └── api.py
├── frontend/
│   └── streamlit_app.py
├── eval/
│   ├── golden_queries.json
│   ├── metrics.py
│   └── run_eval.py
├── tests/
├── configs/
└── Dockerfile
```

## Этап 1. Ingestion (3-5 дней)

### Multi-modal indexing

```python
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='ru')
clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = SentenceTransformer('BAAI/bge-m3')

def ingest_document(image_path, metadata):
    image = Image.open(image_path)
    
    # 1. OCR text
    ocr_result = ocr.ocr(image_path)
    text = ' '.join([line[1][0] for box in ocr_result for line in box])
    
    # 2. Image embedding
    img_inputs = clip_processor(images=image, return_tensors="pt")
    img_emb = clip.get_image_features(**img_inputs).detach().numpy()[0]
    
    # 3. Text embedding (если есть)
    text_emb = text_encoder.encode(text) if text else None
    
    # 4. VLM для structured metadata (optional)
    description = vlm.describe(image)
    
    # 5. Save to vector DB
    qdrant.upsert(
        collection_name="documents",
        points=[{
            "id": doc_id,
            "vector": {"image": img_emb.tolist(), "text": text_emb.tolist()},
            "payload": {"path": image_path, "text": text, "description": description, ...}
        }]
    )
```

### Batch processing

Для большого корпуса (10K+) — async pipeline с queue (Celery + Redis).

## Этап 2. Search (2-3 дня)

### Hybrid search

```python
def search(query: str | Image | both, top_k=10):
    candidates = []
    
    # Text → image search (CLIP)
    if query.text:
        text_emb_for_image_search = clip.get_text_features(query.text)
        results_clip = qdrant.search(
            collection_name="documents",
            query_vector=("image", text_emb_for_image_search),
            limit=50
        )
        candidates.extend(results_clip)
    
    # Text → text search (BM25 + dense)
    if query.text:
        text_emb = text_encoder.encode(query.text)
        results_text = qdrant.search(
            collection_name="documents",
            query_vector=("text", text_emb),
            limit=50
        )
        candidates.extend(results_text)
    
    # Image → image search
    if query.image:
        img_emb = clip.get_image_features(query.image)
        results_image = qdrant.search(...)
        candidates.extend(results_image)
    
    # Dedup, RRF fusion
    fused = rrf_fusion(candidates)
    
    # Rerank with cross-encoder
    reranked = cross_encoder.rerank(query, fused[:50])
    
    return reranked[:top_k]
```

### Reranking

Используем cross-encoder (например, BAAI/bge-reranker-v2-m3) для top-50 → top-10.

## Этап 3. Answer generation (1-2 дня)

```python
def answer_query(query, retrieved):
    # Format context with images
    context_text = ""
    images = []
    for item in retrieved[:5]:
        context_text += f"Source [{item.id}]: {item.text[:500]}\n"
        images.append(item.image)
    
    # VLM answer
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in images],
            {"type": "text", "text": f"Based on these documents and images, answer:\n{query}\n\nContext:\n{context_text}\n\nCite sources by ID."}
        ]
    }]
    answer = vlm.generate(messages)
    return answer
```

## Этап 4. Eval (1-2 дня)

### Golden dataset

50-100 queries с эталонными ответами и нужными документами.

```json
[
  {
    "query": "Какие модели на солнечной энергии есть в каталоге?",
    "expected_docs": ["doc_42", "doc_87", "doc_103"],
    "expected_answer_keywords": ["solar", "panel", "model X"]
  }
]
```

### Метрики

- **Retrieval:** Recall@5, MRR@10.
- **Generation:** BLEU/ROUGE с эталоном (приблизительно), faithfulness через LLM-as-judge.
- **End-to-end:** thumbs up/down rate на real queries.

## Этап 5. UI и deployment (2-3 дня)

### Streamlit app

```python
import streamlit as st

query = st.text_input("Search...")
image = st.file_uploader("Or upload image", type=['png', 'jpg'])

if st.button("Search"):
    results = search(query=query, image=image)
    answer = answer_query(query, results)
    
    st.write(answer)
    
    st.subheader("Sources:")
    for r in results[:5]:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r.image_path, width=200)
        with col2:
            st.write(r.text[:300])
```

### Deployment

- HF Spaces для demo (с moderately-sized DB).
- Self-host через Docker Compose (Qdrant + app + Streamlit).

## Чек-лист готовности

- [ ] Ingested 1000+ изображений / документов.
- [ ] Vector DB с multi-modal embeddings работает.
- [ ] Hybrid search (image+text) даёт релевантные top-10.
- [ ] Reranking повышает Recall@5 на > 10%.
- [ ] VLM генерирует осмысленные ответы с citations.
- [ ] Eval на 50+ golden queries.
- [ ] Streamlit UI работает interactive.
- [ ] Docker Compose stack запускается.
- [ ] README с архитектурой, метриками, demo video.

## Что показать на собесе

1. **Demo:** реальный search и answer с цитатами.
2. **Архитектура:** показать data flow и trade-offs.
3. **Eval rigor:** golden dataset, метрики до/после reranking.
4. **Failure modes:** где система валится и почему.
5. **Scaling:** как растёт latency и cost с 10K → 1M документов.
6. **Тuning:** что бы поменяли при больше времени.

## Идеи усиления

- **Multilingual:** русский + английский поиск.
- **Personalization:** учитывать user history.
- **Active learning:** human feedback на ответы → fine-tune.
- **Streaming answers:** word-by-word output.
- **Caching:** semantic cache для популярных queries.

---

## Поздравляем!

Если вы дошли до этой точки и сделали все 3 капстона — у вас:
- **3 production-ready проекта** в портфолио.
- **Глубокое понимание CV-стека** от пикселей до RAG.
- **Опыт ML в проде:** монitoring, deploy, eval.
- **Готовность к собесам уровня middle/senior CV engineer.**

➡️ Возвращайтесь к [главному README](../README.md) репозитория для других курсов.
