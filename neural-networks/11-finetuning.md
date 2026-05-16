# Урок 11. Fine-tuning и transfer learning

> Цель урока: научиться брать готовую предобученную модель и адаптировать её под свою задачу. Освоить LoRA — основной метод эффективного fine-tuning LLM.

## Зачем fine-tuning

Обучать большую модель с нуля могут позволить себе 5 компаний в мире. Все остальные используют **transfer learning** — берут чужую модель и адаптируют её. Это:

- В 10-1000 раз дешевле.
- Требует на 2-3 порядка меньше данных.
- Часто даёт результат лучше, чем модель с нуля на этих же данных.

## Виды transfer learning

**1. Feature extraction.** Замораживаем всю предобученную модель, используем её как «извлекатель признаков», обучаем только новую голову.

```python
import torch.nn as nn
from torchvision import models

resnet = models.resnet50(pretrained=True)
for p in resnet.parameters():
    p.requires_grad = False
resnet.fc = nn.Linear(2048, 10)  # 10 классов своих
# обучаем только resnet.fc.parameters()
```

**2. Full fine-tuning.** Размораживаем всю модель, обучаем все параметры с **маленьким** learning rate (`1e-5` для трансформеров). Лучшее качество, но дорого.

**3. Discriminative fine-tuning.** Разные lr для разных слоёв: меньший — для нижних (общие фичи), больший — для верхних (специфика задачи).

**4. PEFT (Parameter-Efficient Fine-Tuning).** Обновляем не все веса, а маленький набор дополнительных параметров. Главный представитель — LoRA.

## LoRA: математика и интуиция

**LoRA (Low-Rank Adaptation, 2021)** заметил, что обновления весов при fine-tuning часто имеют **низкий ранг**. Значит, можно представить обновление как произведение двух маленьких матриц:

$$
W_{\text{new}} = W_{\text{frozen}} + \Delta W, \quad \Delta W = B \cdot A
$$

где `A` — матрица `r × k`, `B` — матрица `d × r`, и `r ≪ min(d, k)` (обычно r = 4-64).

Параметров вместо `d·k` становится `r(d+k)` — **в десятки раз меньше**. `W_frozen` не обучается, обучаются только `A` и `B`.

**Что это даёт:**

- Можно fine-tune 7B модель на одной 24GB GPU.
- LoRA-веса весят 5-200 MB вместо 14 GB — удобно хранить много адаптеров под разные задачи.
- На inference можно «склеить» `W + BA` обратно — нулевые накладные расходы.

```python
# Псевдокод LoRA-обёртки
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```

Заметьте инициализацию: `B` нулями, `A` маленькими случайными. В начале обучения `BA = 0` → модель идентична оригинальной. Это **критично** для стабильности.

## Готовые инструменты: HuggingFace + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_4bit=True)
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # обычно достаточно
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable: ~10M из ~7000M — 0.15%
```

С квантизацией в 4 бита (`load_in_4bit=True`) это **QLoRA** — можно fine-tune Mistral 7B даже на бесплатной T4 в Colab.

## Когда какой подход

| Ситуация | Что использовать |
|---|---|
| 100-1000 примеров своих данных | Feature extraction или LoRA |
| 1000-10 000 примеров | LoRA |
| 10 000-100 000 примеров | LoRA с большим `r` или full fine-tune |
| 100 000+ примеров и большой бюджет | Full fine-tune |
| Нужно много задач на одной модели | LoRA адаптеры (по адаптеру на задачу) |

## SFT и instruction-tuning

Современные LLM проходят 3 стадии:

1. **Pretraining** — на огромном корпусе текста. Это «знания».
2. **SFT (Supervised Fine-Tuning)** — на парах «инструкция → ответ». Это «умение следовать инструкциям».
3. **RLHF / DPO** — выравнивание под человеческие предпочтения.

Большинство практических задач решается на этапе 2: возьмите Mistral/Llama, соберите 1-10k примеров под вашу задачу (например, «вопрос по нашему API → структурированный ответ»), запустите LoRA-SFT. Часто этого достаточно, чтобы обогнать GPT-4 на узкой задаче.

## Главные грабли

- **Catastrophic forgetting.** При full fine-tuning модель может забыть общие знания. Решения: меньший lr, LoRA, замораживание нижних слоёв.
- **Утечка данных.** Если test-данные похожи на train — метрики будут наврут. Делите по времени или по пользователям, не случайно.
- **Слишком много эпох.** Для fine-tuning большой модели обычно достаточно 1-3 эпох. Дальше — переобучение.
- **Неправильный формат данных.** Для instruction-tuning важно точно соблюдать chat template модели. Mistral, Llama-3, Qwen имеют **разные** форматы.

## 8 практических заданий

1. **Feature extraction.** Возьмите ResNet50, замените голову на классификатор 5 цветов (`flowers-recognition` с Kaggle). Обучите только голову. Получите accuracy > 90%.
2. **Discriminative fine-tuning.** Размораживайте слои от верхних к нижним по эпохам. Сравните финальную accuracy с задание 1.
3. **LoRA с нуля.** Реализуйте `LoRALinear` и оберните им `nn.Linear` в маленьком трансформере (urok 09). Покажите, что обучается только LoRA.
4. **BERT fine-tune.** Возьмите `distilbert-base-multilingual-cased`, дообучите на русскоязычной классификации тональности (RuSentiment). Получите accuracy > 80%.
5. **QLoRA на Mistral.** Дообучите Mistral 7B на 1000 парах «вопрос-ответ» по любой узкой теме. Сгенерируйте 5 ответов до и после fine-tuning, сравните.
6. **Адаптер-зоопарк.** Обучите 2 разных LoRA-адаптера на одной базовой модели (например, «отвечать как Достоевский» и «отвечать как технарь»). Переключайте их при инференсе.
7. **Catastrophic forgetting.** Дообучите Mistral на узкой задаче с большим lr (`5e-4`). Проверьте, как просел его MMLU-score на общих вопросах. Сравните с малым lr (`5e-5`).
8. **Свой портфолио-проект.** Соберите датасет из 500 примеров своей реальной задачи. Дообучите LoRA-адаптер. Выложите на HuggingFace Hub. Напишите README.

## Чек-лист урока

- [ ] Я понимаю разницу между feature extraction и full fine-tune.
- [ ] Я могу руками вывести параметры LoRA-обновления.
- [ ] Я обучил LoRA-адаптер на готовой LLM.
- [ ] Я знаю, что такое QLoRA и зачем 4-битная квантизация.
- [ ] Я понимаю, как делятся pretraining / SFT / RLHF.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 10](./10-embeddings-tokenization.md) · [README курса](./README.md) · ▶︎ [Урок 12 — От модели к продакшну](./12-production.md)
