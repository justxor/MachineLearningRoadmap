# 06. Классификация изображений

Самая базовая CV-задача и самая распространённая в проде. От «горячо/холодно» (бинарная) до тысяч классов с иерархией.

## Бейзлайн за час

```python
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

model = timm.create_model('convnext_small.fb_in22k_ft_in1k', 
                          pretrained=True, num_classes=NUM_CLASSES)
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, total_steps=len(train_loader) * EPOCHS,
    pct_start=0.1
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch}: {correct/total:.4f}")
```

Этот код — реальный production-ready бейзлайн. На большинстве задач он даст приличный результат, который потом можно улучшать.

## Label Smoothing

Вместо one-hot `[0, 0, 1, 0, 0]` — `[0.02, 0.02, 0.92, 0.02, 0.02]`. Меньше overconfidence, лучше calibration, +0.5-1% accuracy.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Class imbalance

### Class weights в loss

```python
counts = torch.bincount(torch.tensor(train_labels))
weights = 1.0 / counts.float()
weights = weights / weights.sum() * len(counts)
criterion = nn.CrossEntropyLoss(weight=weights.cuda())
```

### WeightedRandomSampler

Балансирует на уровне sampling:

```python
from torch.utils.data import WeightedRandomSampler
weights = 1.0 / counts[train_labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Focal Loss

Для сильного дисбаланса (детекция, OCR):

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()
```

## Multi-label classification

Картинке может быть присвоено **несколько** меток. Например, фото может быть {animal, dog, outdoor, sunny}.

```python
# Head: один sigmoid на каждый класс, не softmax!
model = timm.create_model('convnext_small', pretrained=True, num_classes=NUM_CLASSES)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Targets: multi-hot encoding
y = torch.zeros(NUM_CLASSES)
y[[2, 5, 7]] = 1  # классы 2, 5, 7 присутствуют
```

**Метрики:** mAP per class, mean F1, hamming loss.

## Hierarchical classification

Если есть иерархия (Cat ⊂ Animal ⊂ Living):

**Подходы:**
1. **Flat:** просто предсказываем leaf класс. Простой, но игнорирует структуру.
2. **Hierarchical softmax:** softmax на каждом уровне иерархии.
3. **Loss с штрафами:** ошибка на близких классах (cat vs dog) штрафуется меньше, чем на далёких (cat vs car).

## Self-supervised pretraining

Если своих labeled данных мало, но **много** unlabeled — можно pretrain через self-supervised:

- **MAE (Masked AutoEncoder):** маскируем 75% патчей, учимся восстанавливать. Mask Vision Transformer (MAE-ViT) — стандарт.
- **DINO/DINOv2:** student-teacher с разными аугментациями.
- **SimCLR / MoCo:** contrastive learning.

**Когда:** есть 100K-1M unlabeled, и < 10K labeled. MAE/DINO pretrain → fine-tune на labeled.

## Few-shot learning

Сценарии когда у вас 5-50 примеров на класс:

1. **Linear probing на CLIP/DINOv2:** обучить только linear head поверх замороженных features.
2. **Prompt-tuning CLIP:** подобрать текстовые промпты, не fine-tunить веса.
3. **k-NN classification на embeddings:** простой, но рабочий бейзлайн.

```python
# k-NN на DINOv2 features
features = dinov2(images)  # [N, 1024]
similarities = features @ query_features.T
predictions = labels[similarities.topk(k=5, dim=0).indices].mode().values
```

Часто бьёт fine-tune на маленьких датасетах.

## OOD detection

В проде модель встречает классы, которых не было в train. Что делать?

**Подходы:**
- **Max softmax:** если confidence низкая → OOD. Самый простой.
- **Energy-based:** `-logsumexp(logits)`. Лучше max softmax.
- **MSP (Maximum Softmax Probability) + temperature scaling.**
- **Mahalanobis distance** в feature space.
- **Train с outlier exposure** (если есть proxy OOD данные).

## Калибровка

После training модели обычно переуверены. Калибруем temperature на val set:

```python
# Найти T, минимизирующее NLL на val
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.T

# Обучить T (один скаляр) на val labels
```

## Train/val split: best practices

- **Stratified** по классам.
- **Group split** если есть зависимости (один пользователь не должен быть в train и val).
- **Не leak'ить дубликаты:** `imagehash` для проверки.
- **OOD-eval set:** отдельная выборка с реалистичными edge-cases.

## Production checklist

- [ ] Latency измерена на target hardware.
- [ ] Confidence калибрована.
- [ ] OOD detection включён.
- [ ] Per-class metrics на val (не только overall).
- [ ] Bias по сегментам (sex, age, ethnicity) — для regulated industries.
- [ ] Adversarial robustness тест (FGSM, PGD).
- [ ] Smoke test pipeline (10 батчей перед длинным train).

## Антипаттерны

- **Softmax для multi-label** (нужен sigmoid).
- **Несбалансированный val при дисбалансе train.** Метрика на val будет вводить в заблуждение.
- **Использовать accuracy при дисбалансе.** Лучше F1, mAP.
- **Не считать per-class metrics.** Overall accuracy 95% может скрывать «класс X имеет precision 10%».
- **Train/val leak.** Дубликаты в обеих частях → overfit на тест.

## Задания

1. На своём датасете (1000+ картинок, 5+ классов) натренируйте бейзлайн. Цель: > 80% top-1.
2. Реализуйте class-balanced training через WeightedRandomSampler. Сравните с обычным.
3. Добавьте label smoothing, EMA, MixUp. Замерьте gain от каждого по отдельности.
4. Сравните **fine-tune ConvNeXt** vs **linear probe DINOv2** на маленьком датасете (200 примеров).
5. Реализуйте OOD detection через max softmax confidence. Постройте ROC-кривую in-vs-OOD.
6. Калибруйте модель через temperature scaling. Сравните ECE до и после.

## Чек-лист

- [ ] Знаю production-ready бейзлайн.
- [ ] Умею работать с class imbalance.
- [ ] Отличаю multi-class и multi-label.
- [ ] Применяю label smoothing.
- [ ] Знаю OOD detection.
- [ ] Калибрую confidence перед deploy.

## Дальше

➡️ [07-detection.md](./07-detection.md) — детекция объектов.
