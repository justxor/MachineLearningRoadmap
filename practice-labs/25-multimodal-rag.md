# 🌈 Лаба 25: Multimodal RAG (текст + картинки) 🔴

## Цель

Построить RAG-систему, которая работает с PDF/презентациями, включая изображения, графики и таблицы. Ответы включают цитаты и визуальные источники.

## Стек

- VLM: GPT-4o, Claude 3.5 Sonnet, Llava, Qwen-VL.
- Image embeddings: CLIP / SigLIP.
- Text embeddings: bge / e5.
- Vector store: Qdrant / Weaviate.
- PDF parsing: unstructured / pdfplumber / pix2struct.

## Минимальный пайплайн

1. Парсинг: извлечь текст + изображения из PDF.
2. Для каждого изображения: VLM генерирует caption.
3. Индексируем текстовые chunks + image captions.
4. Query → retrieval топ-k chunks + relevant images.
5. VLM-prompt: текст + images → ответ с цитатами.
6. UI: ответ + показать источники (картинка + страница).

## Код: image captioning (Claude)

```python
import anthropic, base64
client = anthropic.Anthropic()

def caption(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":b64}},
            {"type":"text","text":"Опиши в 5–7 предложениях, что на картинке."}
        ]}],
    )
    return msg.content[0].text
```

## Метрики

- Recall@k для текстовых и image chunks.
- Faithfulness (ragas / LLM-judge).
- Image relevance (правильные картинки в ответе).
- Latency: pdf-parsing, retrieval, VLM.
- Cost per query.

## Расширения

- ColPali — эмбеддинг прямо из страниц PDF как изображений.
- Table extraction (TabularNet / GPT-4o).
- Chart-to-data: VLM извлекает numbers из графика.
- Multi-step reasoning over images.
- Reranker для mixed-modal results.

## Критерии приёмки

- [ ] Индексируются и текст и изображения.
- [ ] VLM отвечает с цитатами на страницы и картинки.
- [ ] Recall@5 >0.75.
- [ ] UI показывает источники.
- [ ] Eval-скрипт в репо.
- [ ] Cost/latency budget прописан.

## Анти-паттерны

- ❌ Индексировать все в OCR-текст (теряется визуальная информация).
- ❌ Изображения без caption — невозможно найти.
- ❌ Отправлять все страницы в VLM (взрыв стоимости).
- ❌ Нет fallback если VLM выбрал нерелевантную картинку.

---

[← Назад к Practice Labs](./README.md)
