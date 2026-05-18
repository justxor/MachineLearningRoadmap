# 🦙 Лаба 15: Fine-tune Llama под доменную задачу (LoRA) 🔴

## Цель

Адаптировать Llama-3-8B под специфическую задачу (юридическая помощь, медицинские Q&A) через LoRA. Понять trade-off память/качество и как валидировать результат.

## Датасет

- От 1k до 50k instruction-response пар в JSONL.
- Открытые: Alpaca, Dolly, OpenAssistant.
- Свой: сгенерировать через GPT-4 + ручная проверка (golden set 200 примеров).

## Минимальный пайплайн

1. Формат данных: chat template (system/user/assistant).
2. Квантование базовой модели (4-bit, bitsandbytes).
3. LoRA-адаптер: r=16, alpha=32, target=q/k/v_proj.
4. SFTTrainer (trl) или axolotl/unsloth.
5. Eval: генерация на hold-out → LLM-as-judge.
6. Экспорт: merge adapter и публикация в HF Hub.

## Код: LoRA (trl)

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj"], task_type="CAUSAL_LM")
cfg = SFTConfig(output_dir="out", num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4, bf16=True)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, tokenizer=tok)
trainer.train()
```

## Метрики

- Validation loss и perplexity.
- Доменные бенчмарки (exact match, BLEU, ROUGE — в зависимости от задачи).
- LLM-as-judge: pairwise comparison с base model.
- Hallucination rate на golden set.

## Расширения

- DPO/ORPO после SFT.
- QLoRA + Flash Attention 2.
- Сравнение r=8/16/64 (бюджет/качество).
- Continued pretraining на доменном корпусе + SFT.
- vLLM для быстрого инференса.

## Критерии приёмки

- [ ] Модель побеждает base на ≥1% pairwise.
- [ ] Нет catastrophic forgetting (проверь на MMLU-subset).
- [ ] Hallucination rate ниже бейзлайна.
- [ ] Adapter опубликован в HF Hub.
- [ ] Демо или инференс-скрипт.

## Анти-паттерны

- ❌ Fine-tuning без ясной задачи («чтобы было лучше»).
- ❌ Маленький датасет (<500) — модель переобучится.
- ❌ Нет valid сета — не видно overfit.
- ❌ Метрики на train, не на hold-out.
- ❌ Игнор base model как baseline.

---

[← Назад к Practice Labs](./README.md)
