# 05. Backbones и transfer learning

В 99% реальных проектов вы **не обучаете CNN с нуля**. Берёте pretrained backbone и адаптируете под задачу.

## Что такое backbone

Backbone — feature extractor: входная картинка → embedding. Над ним вешается «голова» (head) под конкретную задачу:

- Классификация → FC + softmax.
- Детекция → FPN + detection heads.
- Сегментация → decoder (upsampling).
- Embeddings → projection head.

**Принцип:** backbone обучен на огромном датасете (ImageNet, JFT, LAION) и научился извлекать общие визуальные признаки. На вашей задаче дообучается голова (или весь стек).

## Современные backbones (2026)

| Семейство | Размер | Скорость | Когда |
|-----------|--------|----------|-------|
| **ResNet-50/101** | 25M / 45M | Быстро | Baseline, классика, хорошо понятен |
| **EfficientNet-B0...B7** | 5-66M | Сбалансировано | Mobile/edge, хороший trade-off |
| **EfficientNetV2** | 24-120M | Быстрее B-серии | Современный EfficientNet |
| **ConvNeXt-T/S/B/L** | 28-198M | Быстро | Modern CNN, бьёт ResNet и часто ViT |
| **ViT-B/L** | 86-307M | Средне | Когда есть много данных или хороший pretrain |
| **Swin-T/B/L** | 28-197M | Средне | Гибрид: ViT с локальным attention, отлично для детекции |
| **DINOv2** (ViT-B/L) | 86-1.1B | Средне | SOTA self-supervised features, zero-shot отлично |
| **CLIP ViT-L** | 304M | Средне | Multi-modal, zero-shot |
| **MobileNetV3 / MobileViT** | 2-7M | Очень быстро | On-device |

**Правило выбора:**
- Mobile/edge → MobileNetV3, EfficientNet-B0/B1.
- Server, balanced → ConvNeXt-Small или EfficientNetV2-S.
- Server, top quality → ConvNeXt-Large или Swin-Large.
- Multi-modal / zero-shot → DINOv2 или CLIP.

**Где брать:**
```python
import timm
model = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=True)
# 1000+ моделей с разным pretrain
```

`timm` (PyTorch Image Models) — стандарт для CV в 2026. torchvision устаревает.

## Pretrain: на чём важно

| Pretrain | Что | Качество downstream |
|----------|-----|---------------------|
| **ImageNet-1K** | 1.2M картинок, 1000 классов | Baseline |
| **ImageNet-22K** | 14M, 22K классов | +3-5% к downstream |
| **JFT-300M / JFT-3B** | Private Google | Лучший supervised |
| **LAION-2B / LAION-5B** | Web pairs (картинка+текст) | Для CLIP-style |
| **DINOv2** | Self-supervised | Featuring без меток, отлично |
| **MAE** | Masked autoencoder | Self-supervised, основа многих ViT |

**Правило:** если можете выбрать — берите модель с большим/качественным pretrain. Часто `-in22k` суффикс в timm = +pp accuracy на downstream.

## Стратегии transfer learning

### Feature extraction (заморозка backbone)

```python
model = timm.create_model('convnext_small', pretrained=True, num_classes=10)
for p in model.parameters():
    p.requires_grad = False
# Размораживаем только голову
for p in model.head.parameters():
    p.requires_grad = True
```

**Когда:** очень маленький датасет (< 1K), сильно отличается от ImageNet.

**Pros:** быстро, мало overfitting.
**Cons:** потолок качества.

### Fine-tuning всей сети

Размораживаем всё, обучаем с **маленьким** LR для backbone, большим — для головы:

```python
from torch.optim import AdamW

# Discriminative learning rates
params = [
    {'params': model.head.parameters(), 'lr': 1e-3},
    {'params': model.stem.parameters(), 'lr': 1e-5},
    {'params': model.stages.parameters(), 'lr': 1e-4},
]
optimizer = AdamW(params, weight_decay=0.05)
```

**Когда:** средний/большой датасет (5K+).

**Правило:** backbone LR в 10-100 раз меньше LR головы.

### Layer-wise unfreezing (fast.ai стиль)

1. Заморозить всё, обучить голову N эпох.
2. Разморозить последний блок, обучить с маленьким LR.
3. Постепенно размораживать вниз.

Часто даёт лучшее качество, чем «разморозить всё сразу».

### LoRA / Adapters для CV

Современный тренд: вместо full fine-tune добавляем маленькие модули. Особенно полезно для больших моделей (ViT-L, DINOv2).

```python
# Через peft (Hugging Face)
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"])
model = get_peft_model(base_model, config)
```

**Pros:** мало параметров (~1%), можно держать N LoRA для N задач.
**Cons:** иногда чуть ниже качество, чем full fine-tune.

## Выбор размера модели

**Правило большого пальца:**
- < 1K examples → smallest model + feature extraction.
- 1K-10K → small/medium model + fine-tune головы и последних блоков.
- 10K-100K → medium/large model + полный fine-tune.
- 100K+ → large model + полный fine-tune, может с своим pretrain.

**Подвох:** большая модель на маленьком датасете — переобучение, хуже маленькой. **Не «больше = лучше»**.

## Learning rate и расписания

Стандартный train для transfer:

```python
# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Scheduler: warmup + cosine
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps,
                       pct_start=0.1)  # 10% warmup, потом cosine decay
```

**LR Finder** (fast.ai): обучаете с растущим LR от 1e-7 до 1, смотрите, где loss падает быстрее всего. Берёте LR на 1/10 от точки минимума.

## EMA (Exponential Moving Average)

Поддерживаем «теневую» копию весов:

```
ema_weights = decay * ema_weights + (1 - decay) * weights  # decay ≈ 0.9999
```

На eval используем ema_weights — обычно +0.5-1% accuracy. Стандарт для SOTA-моделей.

```python
from timm.utils import ModelEmaV2
ema = ModelEmaV2(model, decay=0.9999)
# В тренировочном цикле:
ema.update(model)
# На eval:
model_for_eval = ema.module
```

## Метрики обучения: что мониторить

- **Train loss / Val loss.** Разрыв = overfit.
- **Val metric.** Главный сигнал. Не loss!
- **LR.** Видно ли warmup/decay по плану.
- **Gradient norm.** Если skyrockets — gradient clipping не работает.
- **Параметры по слоям.** Histograms в TensorBoard — заметите dead neurons.

## Антипаттерны

- **Не использовать pretrained.** Обучаете ResNet50 с нуля на 5K картинок — потеря времени.
- **Тот же LR на backbone и голову.** Backbone «забудет» pretrain, или голова недоучится.
- **Игнорировать timm.** torchvision устарел, в timm в 10 раз больше моделей с лучшими pretrain.
- **Big model on small data.** ViT-L на 1K примеров — переобучение гарантировано.
- **Не использовать warmup для трансформеров.** Нестабильное обучение, NaN в loss.

## Задания

1. Загрузите 5 backbones из timm (ResNet-50, EfficientNet-B0, ConvNeXt-Small, ViT-Base, Swin-T). На своих данных fine-tune-те и сравните: accuracy, latency, memory.
2. Реализуйте discriminative LR: 1e-5 для backbone, 1e-3 для головы. Сравните с одним LR.
3. Попробуйте feature extraction vs full fine-tune на маленьком датасете (1000 примеров). Какой выигрывает?
4. Реализуйте EMA модели. Замерьте разницу в val accuracy с/без EMA.
5. Используйте `timm.optim` оптимизаторы (LAMB, Lookahead) и сравните с AdamW.
6. Возьмите DINOv2 features (без обучения!) → обучите linear classifier поверх. Какое качество vs обучение всего ResNet с нуля?

## Чек-лист

- [ ] Знаю основные backbones и их trade-offs.
- [ ] Использую timm как стандарт.
- [ ] Понимаю стратегии: feature extraction vs full fine-tune.
- [ ] Использую discriminative LR.
- [ ] Применяю warmup + cosine schedule.
- [ ] Знаю про EMA и применяю для production-моделей.

## Дальше

➡️ [06-classification.md](./06-classification.md) — классификация изображений end-to-end.
