# 🖼️ Лаба 06: Классификатор картинок с нуля (PyTorch) 🟢

## Цель

Обучить CNN с нуля, потом fine-tune предобученной модели. Понять полный цикл: DataLoader, аугментации, обучение, валидация, инференс.

## Датасет

- CIFAR-10 (10 классов, 60k картинок 32x32).
- Альтернатива: Tiny ImageNet или свой датасет из папок.

## Минимальный пайплайн

1. Dataset/DataLoader, transforms (Normalize, RandomCrop, HorizontalFlip).
2. Простая CNN: 3-4 свёрточных блока.
3. Обучение: 20 эпох, Adam, CrossEntropy.
4. Fine-tune ResNet-18 (pretrained=True).
5. Confusion matrix + classification report.
6. Инференс на одной картинке + Grad-CAM.

## Код: training loop

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)
```

## Метрики

- Top-1 accuracy, top-5 accuracy.
- Per-class precision/recall.
- Baseline custom CNN: ~75%, fine-tuned ResNet-18: ~92%.

## Расширения

- MixUp / CutMix аугментации.
- Label smoothing.
- Cosine LR scheduler + warmup.
- Test-time augmentation.
- ONNX-экспорт и инференс в ONNX Runtime.

## Критерии приёмки

- [ ] Плоты loss/accuracy по эпохам.
- [ ] Confusion matrix в отчёте.
- [ ] Grad-CAM для 5 примеров.
- [ ] Accuracy custom CNN >70%, fine-tuned >90%.
- [ ] Random seed фиксирован, обучение воспроизводимо.

## Анти-паттерны

- ❌ Отсутствие normalization (модель медленно учится).
- ❌ Аугментации на валидации.
- ❌ Большой LR без warmup — взрыв градиентов.
- ❌ Игнорирование num_workers (боттлнек на IO).

---

[← Назад к Practice Labs](./README.md)
