# 03. Аугментации и data pipelines

Аугментации часто дают больше, чем замена модели на «следующую SOTA». Хороший pipeline — это **+5-15% качества бесплатно**.

## Зачем аугментации

1. **Регуляризация.** Модель видит «новые» примеры → меньше переобучения.
2. **Invariance.** Учим модель быть устойчивой к flip, rotation, lighting.
3. **Класс-баланс.** Aug на редкие классы — мягкая балансировка.
4. **Дешевле сбора данных.** Часто +10K augmented примеров > +1K real.

## Базовые аугментации

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**Правило:** на train — agressive aug. На val/test — только resize и normalize, никакой случайности.

## Albumentations vs torchvision vs Kornia

| Библиотека | Скорость | API | Когда брать |
|------------|----------|-----|-------------|
| **torchvision.transforms.v2** | Средняя | Простой | Если уже в torchvision stack |
| **albumentations** | Быстрая (CPU) | Гибкий, поддержка bbox/keypoints/masks | Дефолт для CV-проектов |
| **Kornia** | Быстрая (GPU) | PyTorch-native | Когда aug делается на GPU вместе с моделью |

**Подвох albumentations:** работает на numpy, не на tensor. ToTensor в конце.

## Аугментации для разных задач

### Классификация

Standard set:
- RandomResizedCrop
- HorizontalFlip (если не зависит от ориентации)
- ColorJitter
- Cutout / RandomErasing
- MixUp / CutMix (на уровне batch)

### Object Detection

**Важно:** bbox должен трансформироваться вместе с картинкой.

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(width=640, height=640),
    A.HueSaturationValue(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

augmented = transform(image=img, bboxes=boxes, class_labels=labels)
```

Не использовать без коррекции:
- Rotation (если только не 90/180/270) — рамка не повернётся правильно.
- Heavy cropping — может вырезать всё.

**Mosaic augmentation** — стандарт в YOLO: 4 картинки склеиваются в одну. Резко поднимает качество.

### Segmentation

Маски должны трансформироваться вместе с картинкой:

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=512, width=512),
    A.Rotate(limit=15, p=0.5),
])
augmented = transform(image=img, mask=mask)
```

**Подвох:** интерполяция масок должна быть **nearest** (не bilinear), иначе появляются «промежуточные» классы. По умолчанию в albumentations всё ок.

### Keypoints / Pose

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
], keypoint_params=A.KeypointParams(format='xy'))
```

**Подвох:** при horizontal flip нужно swap парные keypoints (правое ухо ↔ левое ухо). Albumentations не делает это автоматически.

## Продвинутые техники

### MixUp

Линейная комбинация двух картинок + интерполяция лейблов:

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# В loss:
loss = lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)
```

### CutMix

Вырезание прямоугольника из одной картинки → вставка в другую. Лейблы по площади.

### AutoAugment / RandAugment / TrivialAugment

Автоматический поиск оптимальных aug-политик. **TrivialAugment** — современный дефолт: на каждой картинке случайная одна аугментация с случайной силой. Минимум кода, отличный результат.

```python
from torchvision.transforms.v2 import TrivialAugmentWide
transform = T.Compose([
    T.Resize((256, 256)),
    TrivialAugmentWide(),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])
```

### Specific augmentations для типов данных

- **Медицина:** elastic deformations, gamma correction.
- **Документы:** perspective transform, blur (имитация фото), JPEG compression artifacts.
- **Satellite:** RandomRotate90 (там нет «верха»).
- **Лица:** жёсткие ограничения — нельзя flip ассиметричных лиц, осторожно с color jitter.

## Test-time augmentation (TTA)

На eval/inference прогоняем картинку несколько раз с разными aug → усредняем предсказания. +1-3% accuracy.

```python
def tta_predict(model, img, transforms_list):
    preds = []
    for t in transforms_list:
        x = t(image=img)['image'].unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
        preds.append(p)
    return torch.stack(preds).mean(0)
```

**Цена:** N inference. Используется в Kaggle и в задачах, где качество критичнее latency.

## Data pipeline в проде

### Bottleneck: data loading

Симптом: GPU utilization < 50% при обучении. Решения:

1. **`num_workers > 0`** в DataLoader. Стандарт = 4-8.
2. **`pin_memory=True`** для GPU transfer.
3. **`persistent_workers=True`** — не пересоздавать workers каждую эпоху.
4. **`prefetch_factor=2-4`** — заранее загружать batches.

```python
loader = DataLoader(dataset, batch_size=32, num_workers=8, 
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
```

### Большие датасеты

Тысячи мелких JPEG на диске → IOPS bottleneck.

**Решения:**
- **webdataset:** tar-файлы со streaming.
- **LMDB:** key-value store.
- **HDF5:** хорошо для масок и метаданных.
- **FFCV:** быстрый формат для PyTorch, в разы быстрее обычного.

### GPU augmentations

Если CPU не справляется — переносим aug на GPU через Kornia или nvidia DALI. Освобождает CPU, картинки aug-ятся параллельно с forward pass.

## Антипаттерны

- **Aug на val/test.** Если рандомные aug на val — метрика прыгает, нельзя сравнивать модели.
- **Слишком aggressive aug.** Если картинки уже не похожи на реальные данные — модель учится неправильному.
- **Aug без знания домена.** Horizontal flip медицинских снимков может перепутать лево и право.
- **Игнорировать bbox/mask aug.** Аугментировали картинку, забыли про метки — данные сломаны.

## Задания

1. На CIFAR-10 натренируйте ResNet18 без aug и с базовым набором. Замерьте разницу в accuracy.
2. Реализуйте MixUp с нуля и сравните на small dataset (1000 примеров) — насколько он помогает.
3. Постройте pipeline для object detection с правильной aug bbox'ов через albumentations. Визуализируйте 10 примеров до/после.
4. Сравните скорость training с разным `num_workers` (0, 2, 4, 8). На какой момент перестаёт ускорять?
5. Реализуйте TTA для классификации (5 версий: original + 4 flips/crops). Измерьте +accuracy и время инференса.
6. Конвертируйте свой датасет в формат webdataset или FFCV. Сравните скорость обучения с обычной папкой JPEG.

## Чек-лист

- [ ] Знаю basic aug set для классификации.
- [ ] Умею делать bbox aug для детекции.
- [ ] Умею делать mask aug для сегментации.
- [ ] Понимаю MixUp/CutMix и могу реализовать.
- [ ] Оптимизировал DataLoader для full GPU utilization.

## Дальше

➡️ [04-metrics.md](./04-metrics.md) — метрики CV: accuracy, IoU, mAP.
