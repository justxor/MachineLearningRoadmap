# 04. Метрики CV

Правильная метрика — половина успеха. Неверная — даёт чувство «всё работает», пока модель в проде валится.

## Классификация: основы

См. [interview-prep/01-ml-classic.md](../interview-prep/01-ml-classic.md) — там общие принципы (precision, recall, F1, ROC-AUC).

**CV-специфика:**

- **Top-k accuracy.** Для ImageNet стандарт — top-5 (правильный класс в топ-5 предсказаний).
- **Multi-label.** Каждой картинке несколько меток (mAP per class + average).
- **Hierarchical classification.** Если есть иерархия классов (cat → animal), штраф меньше за ошибку на близких классах.

## Детекция: IoU и mAP

### IoU (Intersection over Union)

Для двух bbox: пересечение / объединение. От 0 до 1.

```python
def iou(box1, box2):
    # boxes: [x1, y1, x2, y2]
    x_inter = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_inter = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    inter = x_inter * y_inter
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0
```

**Стандартные пороги:**
- **0.5:** PASCAL VOC, общий стандарт. «Достаточно правильный» детект.
- **0.75:** строже, для медицины и точных задач.
- **0.5:0.05:0.95:** COCO — усреднение по 10 порогам.

### mAP (mean Average Precision)

**Алгоритм:**
1. Для каждой картинки: получаем предсказания (bbox + class + confidence).
2. Сортируем по confidence.
3. Для каждого предсказания: правильное ли? (IoU > threshold с GT).
4. Строим precision-recall кривую (изменяя порог confidence).
5. AP = площадь под PR-кривой.
6. mAP = усреднение AP по всем классам.

**COCO mAP:** усредняем AP@[0.5:0.05:0.95] — 10 порогов IoU. Более строгая метрика.

**Используйте готовое:** `pycocotools` или `torchmetrics.detection.MeanAveragePrecision`. Свой код почти всегда содержит баги.

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox")
metric.update(preds, targets)
result = metric.compute()
# {'map': ..., 'map_50': ..., 'map_75': ..., 'map_small': ..., ...}
```

## Сегментация

### Per-pixel метрики

- **Pixel accuracy:** доля правильно классифицированных пикселей. Обманчива при дисбалансе (90% пикселей — фон).
- **Mean Pixel Accuracy:** усреднение accuracy по классам.

### IoU для сегментации

Считается так же, как для детекции, но на пикселях:

```python
def iou_segmentation(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union if union > 0 else 0
```

### mIoU (mean IoU)

Усреднение IoU по всем классам. **Главная метрика семантической сегментации.**

```python
def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious)
```

**Подвох:** классы с малой площадью (мелкие объекты) занижают mIoU больше, чем большие. На дисбалансе IoU мелкого класса = 0 если модель его «пропустила».

### Dice (F1 для пикселей)

```
Dice = 2 * intersection / (pred_area + gt_area)
```

Связан с IoU: Dice = 2·IoU / (1 + IoU).

**Когда что:**
- **mIoU:** стандарт для семантической сегментации (PASCAL, Cityscapes).
- **Dice:** медицинская сегментация (более «дружелюбен» к мелким структурам).

## Instance Segmentation

### AP segmentation

То же mAP, но IoU считается на масках, не на bbox. Стандарт в COCO.

### Panoptic Quality (PQ)

Объединяет semantic и instance сегментацию:

```
PQ = (Σ IoU correctly matched) / (TP + 0.5·FP + 0.5·FN)
```

Стандарт для panoptic задач (Cityscapes, COCO Panoptic).

## Keypoints / Pose

### OKS (Object Keypoint Similarity)

Аналог IoU для ключевых точек:

```
OKS = exp(-d² / (2·s²·k²))
```

где d — расстояние между predicted и GT keypoint, s — масштаб объекта, k — per-keypoint constant.

**mAP@OKS:** аналогично mAP, но на OKS thresholds. COCO стандарт.

### PCK (Percentage of Correct Keypoints)

Доля keypoints с distance < threshold (обычно % от размера головы или объекта).

## Tracking

### MOTA / MOTP / IDF1

- **MOTA (Multi-Object Tracking Accuracy):** учитывает FP, FN, ID switches.
- **MOTP:** среднее IoU матчей.
- **IDF1:** F1 на уровне идентификаторов (важно для long-term tracking).
- **HOTA:** новый стандарт, балансирует detection и association.

Готовое: `motmetrics`, `TrackEval`.

## OCR

- **Character Error Rate (CER):** editов на символ. CER = (S+D+I)/N.
- **Word Error Rate (WER):** editов на слово.
- **F1 на полях документа** (для structured OCR like invoices).

## Generation (image generation)

- **FID (Fréchet Inception Distance):** расстояние между распределениями features (Inception) реальных и сгенерированных картинок. Меньше = лучше.
- **IS (Inception Score):** старая метрика, не очень хороша. Не используйте как единственную.
- **CLIP Score:** для text-to-image, similarity между CLIP embedding промпта и картинки.
- **LPIPS:** perceptual similarity, для super-res, inpainting.

## Confidence calibration

Модели часто **переуверены** (особенно после CE loss). Если используете confidence в downstream — калибруйте.

**ECE (Expected Calibration Error):**

```python
def ece(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece_value = 0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece_value += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return ece_value
```

**Лечение:** temperature scaling — обучаем один скаляр T на val set.

## Beyond accuracy: production metrics

В проде смотрят не только на качество модели:

- **Latency p50/p95/p99.**
- **Throughput (images/sec).**
- **Memory footprint.**
- **Robustness to OOD** (out-of-distribution).
- **Bias по группам** (skin tone, gender, age).
- **Adversarial robustness.**

## Антипаттерны

- **Accuracy на дисбалансе.** «90% accuracy» — модель предсказывает majority class.
- **Single threshold для mAP.** mAP@0.5 — оптимистичная, легко переобучить.
- **Не учитывать class imbalance в mIoU.** Один редкий класс с IoU=0 драматически снижает mIoU.
- **Eval на train-like distribution.** Должен быть OOD-eval (другая камера, освещение).
- **Не калибровать confidence** перед использованием в downstream.

## Задания

1. Реализуйте IoU с нуля для bbox и сравните с torchmetrics на синтетических примерах.
2. Реализуйте mAP@0.5 с нуля. Сравните с pycocotools на маленьком датасете.
3. Постройте precision-recall curve для своей модели детекции при разных confidence thresholds.
4. Реализуйте mIoU и pixel accuracy для семантической сегментации. На каком случае они расходятся?
5. Посчитайте ECE для своей классификации до и после temperature scaling.
6. Возьмите модель сегментации, замерьте mIoU на validation, потом на сильно отличающихся данных (другое освещение, другая камера). Какой gap?

## Чек-лист

- [ ] Понимаю IoU и могу реализовать с нуля.
- [ ] Знаю, как считается mAP и почему COCO mAP@[0.5:0.95] строже.
- [ ] Умею считать mIoU и Dice для сегментации.
- [ ] Знаю про OKS для keypoints.
- [ ] Понимаю calibration и ECE.
- [ ] В проде меряю не только качество, но и latency, robustness, bias.

## Дальше

➡️ [05-backbones.md](./05-backbones.md) — backbones и transfer learning.
