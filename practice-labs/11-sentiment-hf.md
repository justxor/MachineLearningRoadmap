# 💬 Лаба 11: Sentiment-классификатор (HF transformers) 🟢

## Цель

Обучить классификатор тональности отзывов на BERT-подобной модели. Понять весь пайплайн HuggingFace: tokenizer → Trainer → inference.

## Датасет

- IMDB Reviews (50k, binary).
- Русский: RuReviews или Kinopoisk reviews.
- Альтернатива: Twitter Sentiment140.

## Минимальный пайплайн

1. Загрузка датасета через datasets.
2. Токенизация (distilbert-base-uncased).
3. AutoModelForSequenceClassification.
4. Trainer + TrainingArguments.
5. Эвал: accuracy, F1, confusion matrix.
6. Пуш модели в HF Hub.

## Код: обучение

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=256), batched=True)

args = TrainingArguments("out", num_train_epochs=2, per_device_train_batch_size=16, eval_strategy="epoch")
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
```

## Метрики

- Accuracy, F1-macro.
- ROC-AUC.
- Confusion matrix.
- Baseline TF-IDF + LR: ~88% acc, DistilBERT: ~92%+.

## Расширения

- Multi-class (1-5 star rating).
- Aspect-based sentiment (aspect terms + polarity).
- ONNX экспорт, quantization int8.
- LoRA-fine-tuning для экономии памяти.
- Gradio-демо для портфолио.
- Cross-lingual: обучение EN, тест RU с mBERT/XLM-R.

## Критерии приёмки

- [ ] Accuracy >90% на test.
- [ ] Confusion matrix в отчёте.
- [ ] 5 error cases разобраны.
- [ ] Gradio-демо работает.
- [ ] Inference скрипт в .py.

## Анти-паттерны

- ❌ Padding=max_length без truncation (падает память).
- ❌ Обучение fp32 на GPU вместо fp16/bf16.
- ❌ Игнор длинных текстов (обрезаются, важный сигнал теряется).
- ❌ Сравнение без baseline TF-IDF (не ясно, стоит ли BERT).

---

[← Назад к Practice Labs](./README.md)
