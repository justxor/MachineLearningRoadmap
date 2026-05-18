# 14. VLM: vision-language models

VLM = модели, принимающие на вход картинку + текст, выдающие текст. От image captioning до Visual QA и understand-anything моделей.

## Архитектура VLM

Стандартная схема:

```
[image] → vision encoder (ViT/CLIP) → projector → [image tokens] ─┐
                                                                   ├→ LLM → text out
[text prompt] → tokenizer ─────────────────────→ [text tokens] ───┘
```

Vision encoder обычно frozen, projector — маленький MLP, LLM — fine-tuned для visual instructions.

## Семейство моделей

| Модель | Год | Особенность |
|--------|-----|-------------|
| **LLaVA** | 2023 | Open-source pioneer, CLIP + Vicuna |
| **LLaVA-1.5 / 1.6 / NeXT** | 2023-24 | Улучшения качества, high-res |
| **Qwen-VL / Qwen2-VL** | 2024 | Multilingual, отличный OCR, видео |
| **InternVL** | 2024 | Open-source SOTA, разные размеры |
| **Florence-2** | 2024 | Маленький (0.7B), универсальный |
| **CogVLM** | 2023 | Visual expert module |
| **GPT-4V/4o, Claude 3.5, Gemini 1.5** | 2024 | Closed source SOTA |

В 2026 для open-source производство: **Qwen2-VL** или **InternVL**.

## Использование Qwen2-VL

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "photo.jpg"},
        {"type": "text", "text": "Describe what you see in detail."},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=512)
```

## Применения VLM в проде

### 1. Document understanding

OCR + understanding: понять структуру документа, извлечь поля.

- Qwen2-VL, InternVL — отлично работают.
- **Donut, LayoutLMv3** — специализированные для документов.

### 2. Visual QA

Ответ на вопрос о картинке. Customer support: «что не так с этим товаром?»

### 3. Auto-captioning

Генерация описаний для фото. Accessibility, e-commerce descriptions, social media.

### 4. Content moderation

Не просто «есть ли NSFW», а с пояснением: почему этот контент проблематичен.

### 5. Visual reasoning

Анализ графиков, таблиц, схем. Чтение чертежей, диаграмм UML.

### 6. Embodied AI / Robotics

VLM как «глаза» для робота: «найди красную чашку и принеси».

## Что VLM умеет плохо

- **Точная локализация.** «Где именно?» — координаты часто неточны.
- **Подсчёт.** «Сколько объектов?» — часто ошибается на 5+ объектов.
- **Мелкие детали.** Низкое разрешение vision encoder.
- **OCR редких алфавитов и рукописного.** Только лучшие модели (Qwen2-VL, GPT-4o).
- **Сложные пространственные отношения.** «Что левее чем X?»

## Fine-tuning VLM

Если базовая модель не справляется с domain:

- **LoRA на LLM** — самый дешёвый, часто достаточно.
- **Fine-tune projector** — если visual gap значимый.
- **Fine-tune vision encoder** — только когда совсем плохо.

Стандарт: LLaVA-style training с domain-specific instructions (image, instruction → response).

## Серверная экономика

VLM модели большие (7B-70B+). Cost:

- **Self-hosted Qwen2-VL-7B:** ~$0.0001/1K tokens на A100.
- **GPT-4o API:** ~$5/1M input tokens.
- **Claude 3.5 Sonnet API:** ~$3/1M input tokens.

Для thousands of QPS обычно self-hosted дешевле. Для прототипа — API.

## Подключение VLM в LLM-pipeline

VLM удобно встраивать в **agent**:

```python
@tool
def analyze_image(image_url: str, question: str) -> str:
    """Ask a question about an image."""
    return vlm.predict(image=image_url, prompt=question)
```

Затем агент (Claude, GPT) использует tool when needs to «see».

## OCR через VLM vs специализированные

- **Стандартный OCR (Tesseract, PaddleOCR):** дёшево, быстро, для печатного текста — отлично.
- **VLM (Qwen2-VL, GPT-4V):** дороже, медленнее, но понимает контекст, таблицы, нестандартный layout.

**Гибрид:** OCR → если структура сложная, VLM для understanding.

## Антипаттерны

- **Использовать GPT-4V для всего.** Дорого. Для простого OCR/classification — overkill.
- **Доверять VLM точную локализацию.** Используйте детектор + VLM для понимания.
- **Игнорировать prompt design.** VLM ещё сильнее зависят от промпта, чем LLM.
- **VLM-as-a-judge без калибровки.** Хорошо для творческих задач, плохо для строгих метрик.

## Задания

1. Использовать Qwen2-VL для анализа графика matplotlib. Spotcheck качество.
2. Сравнить Qwen2-VL-7B, InternVL-8B, LLaVA-1.6-7B на VQA задаче.
3. Реализовать AutoCaption для фотобанка: 1000 фоток → описания на русском.
4. Использовать VLM в LangChain агенте: модель может «увидеть» картинку по запросу.
5. Сравнить точность счёта объектов через VLM и через детектор + counter. На каком количестве VLM начинает ошибаться?
6. Fine-tune LLaVA через LoRA на узкой задаче (например, классификация патологий). Сравнить с zero-shot.

## Чек-лист

- [ ] Понимаю архитектуру VLM (vision encoder + projector + LLM).
- [ ] Умею использовать Qwen2-VL/InternVL для задач.
- [ ] Знаю их слабости (counting, precise locations).
- [ ] Понимаю trade-offs self-host vs API.

## Дальше

➡️ [15-generative-cv.md](./15-generative-cv.md) — генеративные модели для CV.
