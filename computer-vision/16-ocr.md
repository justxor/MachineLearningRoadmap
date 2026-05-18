# 16. OCR и распознавание документов

OCR — отдельная индустрия, далеко не только Tesseract. От квитанций до научных статей с формулами — разные подходы.

## Уровни задач

1. **Plain text OCR.** Просто текст с картинки. Sells receipts, books.
2. **Document layout analysis.** Понять структуру: заголовки, абзацы, таблицы.
3. **Form parsing.** Извлечь key-value (счёт-фактура, паспорт).
4. **Handwritten OCR.** Рукописный текст.
5. **Math / formula OCR.** Распознавание формул в LaTeX.

## Open-source стек

### Tesseract

Старичок, ещё работает. Хорош для печатного текста на чистом фоне.

```python
import pytesseract
text = pytesseract.image_to_string(image, lang='rus+eng')
```

**Когда:** прототип, простые сценарии. **Не для:** complex layout, handwriting.

### EasyOCR

PyTorch-based, поддерживает 80+ языков. Качество выше Tesseract на нестандартных шрифтах.

```python
import easyocr
reader = easyocr.Reader(['ru', 'en'])
result = reader.readtext('image.jpg')
```

### PaddleOCR (стандарт 2026)

Лучший open-source OCR. Detection + recognition + structure parsing. Поддержка русского, мульти-язык, документы.

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ru')
result = ocr.ocr('image.jpg')
```

### Donut

End-to-end document understanding без OCR step. Картинка → JSON.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
# Output: structured JSON
```

### LayoutLMv3 / DocFormer

Multi-modal модели для document understanding: text + layout + image. Лучшее качество для structured docs.

## VLM для OCR

Современный тренд: использовать **Qwen2-VL, GPT-4o, Claude 3.5** для OCR. Понимают layout, таблицы, нестандартные документы.

```python
# Qwen2-VL для извлечения данных из счёт-фактуры
prompt = "Extract invoice number, date, total amount as JSON"
result = qwen2vl(image=invoice, prompt=prompt)
```

**Pros:** не нужно train, понимает context. **Cons:** дорого, медленно vs специализированные.

## Pipeline document understanding

```
[scan/photo] → 
preprocessing (deskew, denoise, contrast) →
text detection (где текст) →
text recognition (что написано) →
layout analysis (структура) →
extraction (key-value pairs) →
post-processing (validation, normalization)
```

## Preprocessing для OCR

Критично для качества:

```python
# 1. Серый
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Denoise
denoised = cv2.fastNlMeansDenoising(gray)

# 3. Deskew (выровнять наклон)
# через Hough transform или minAreaRect

# 4. Binarize
binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# 5. Морфология (по нужде)
```

## Таблицы

Сложная задача. Подходы:
- **Detection + structure recognition:** PaddleOCR table, Table Transformer.
- **VLM:** Qwen2-VL умеет читать таблицы в Markdown.
- **End-to-end:** Donut, LayoutLM для document understanding.

## Метрики

- **CER (Character Error Rate):** доля ошибок на символ. Стандарт.
- **WER (Word Error Rate):** на слово.
- **Field-level F1:** для structured extraction (правильно ли извлечено поле).
- **End-to-end accuracy:** доля документов, обработанных полностью корректно.

## Multilingual

- **PaddleOCR:** 80+ языков, отличный русский.
- **Tesseract:** много языков, но качество варьируется.
- **EasyOCR:** 80+ языков.

Для **CJK (китайский, японский, корейский)** PaddleOCR лучший open-source.

## Production scenarios

| Сценарий | Стек |
|----------|------|
| Простой OCR (печатный текст) | **PaddleOCR** |
| Счёт-фактуры | **Donut/LayoutLMv3 + custom fine-tune** или **VLM** |
| Паспорта/IDs | **Detection (YOLO) → CRNN recognition** на собственных данных |
| Чеки | **Donut (CORD pre-trained) + fine-tune** |
| Формулы | **pix2tex (LaTeX-OCR)** |
| Рукописный | **TrOCR, GOT-OCR2.0** |
| Любые документы | **Qwen2-VL или GPT-4o** для сложных, PaddleOCR для massive scale |

## Custom training: когда нужно

Pretrained моделей обычно хватает для general OCR. Custom train нужен:
- Специфический шрифт (готический, древний).
- Низкое качество (старые сканы, размытые фото).
- Узкий домен (медицинские записи).
- Жёсткий privacy (нельзя send в API).

**Подход:** TrOCR architecture (encoder-decoder, ViT + RoBERTa) + fine-tune на ваших данных.

## Cost analysis

- **PaddleOCR self-hosted:** ~$0.00001/page.
- **AWS Textract:** $1.50/1000 pages.
- **GPT-4o vision:** ~$0.005/page.
- **Qwen2-VL self-hosted:** ~$0.001/page.

При scale (миллионы pages в день) — self-host обязателен.

## Антипаттерны

- **Tesseract без preprocessing** на низкокачественных сканах → 30% accuracy.
- **GPT-4o для миллионов pages** — счёт за месяц шокирует.
- **Игнорировать post-processing.** Numbers/dates/emails/etc — нужны regexp validators.
- **Не маркировать confidence.** Передавать downstream без знания, надёжен ли каждый field.

## Задания

1. Распознать 100 чеков (датасеты на kaggle): сравнить Tesseract, PaddleOCR, Donut, GPT-4V.
2. Препроцессинг pipeline для старых сканов: deskew + denoise + binarize. Замерить gain.
3. Fine-tune Donut на свой формат документов (можно сгенерировать через template).
4. Извлечь structured data (invoice number, date, total) из ассорти invoices через VLM с structured outputs.
5. Реализовать confidence threshold: документы с низким confidence → human review.
6. Сравнить cost/quality для batch processing 10K документов: self-hosted PaddleOCR vs API.

## Чек-лист

- [ ] Знаю open-source стек: Tesseract/EasyOCR/PaddleOCR.
- [ ] Понимаю preprocessing для OCR.
- [ ] Знаю Donut, LayoutLM для structured docs.
- [ ] Умею использовать VLM для нестандартных документов.
- [ ] Знаю CER/WER метрики.

## Дальше

➡️ [17-video-analytics.md](./17-video-analytics.md) — видео-аналитика.
