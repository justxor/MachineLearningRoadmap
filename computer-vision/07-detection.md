# 07. Детекция объектов

Object detection = найти объекты на картинке + bbox + class + confidence. После классификации — самая распространённая задача в проде.

## Семейства моделей

### Two-stage (Faster R-CNN family)

1. Region Proposal Network (RPN) предлагает кандидатов.
2. Classifier + regressor на каждый proposal.

**Pros:** высокое качество. **Cons:** медленно. **2026:** морально устарели, кроме Mask R-CNN для instance segmentation.

### One-stage anchor-based (YOLO v3-v8, RetinaNet)

Картинка → grid cells → каждая cell предсказывает N anchors → (offsets + class + confidence).

**YOLO семейство:**
- **YOLOv8 (ultralytics):** стабильный, отличный quality/speed trade-off.
- **YOLOv9, YOLOv10, YOLOv11:** improvements год от года.
- **YOLO-NAS, YOLO-World:** open-vocabulary детекция.

```python
from ultralytics import YOLO

# Train
model = YOLO('yolov8m.pt')  # pretrained на COCO
model.train(data='dataset.yaml', epochs=100, imgsz=640, batch=16)

# Inference
results = model('image.jpg')
```

### One-stage anchor-free (CenterNet, FCOS)

Без anchors — каждая точка grid предсказывает напрямую. Проще, часто быстрее.

### DETR-family (DETR, DINO-DETR, RT-DETR)

Transformer-based, без NMS. Direct set prediction через Hungarian matching.

**RT-DETR (Real-Time DETR):** бьёт YOLO на тех же latency, не нужен NMS. **Современный SOTA для production**.

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r50vd')
model = RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r50vd')
```

## Что выбрать в 2026

| Сценарий | Выбор |
|----------|-------|
| Быстрый старт, embedded, mobile | **YOLOv11-n/s** |
| Production server, баланс | **YOLOv11-m или RT-DETR-R18** |
| Top quality, server | **RT-DETR-R50/R101 или YOLOv11-x** |
| Open-vocabulary (любые классы) | **YOLO-World, Grounding DINO** |
| Instance segmentation | **Mask2Former или YOLOv11-seg** |

## Anchors на пальцах

Anchor — заранее заданный bbox разного размера и aspect ratio. Сеть предсказывает offsets от anchor + objectness + class.

**Зачем:** упрощает обучение (модель учится корректировать готовые boxes, а не предсказывать с нуля).

**Подвох:** anchors должны соответствовать распределению размеров объектов в данных. Если объекты маленькие, а anchors большие — модель плохо работает. Поэтому в YOLO есть автоматический anchor clustering на train set.

## Anchor-free на пальцах

Каждая точка feature map предсказывает: distance to 4 sides bbox + class + centerness (likelihood, что это центр объекта).

**Pros:** меньше hyperparameters, легче переносить на новые домены.
**Cons:** иногда чуть ниже качество.

## NMS (Non-Maximum Suppression)

После forward модель выдаёт сотни кандидатов на один объект. NMS оставляет лучший:

1. Сортируем по confidence.
2. Берём top-1, добавляем в результат.
3. Удаляем все boxes с IoU > 0.5 с этим.
4. Повторяем.

```python
import torchvision.ops as ops
keep = ops.nms(boxes, scores, iou_threshold=0.5)
```

**Варианты:**
- **Soft-NMS:** не удаляет, а понижает score. Лучше для overlapping objects.
- **Class-aware NMS:** делаем NMS отдельно по каждому классу.
- **DETR:** без NMS вообще, прямое set prediction.

## Loss-функции

Стандарт для bbox regression:
- **GIoU / DIoU / CIoU loss** — учитывают geometry, лучше L1/L2.
- **L1 / Smooth L1** — простой, в DETR.

Для classification:
- **Focal Loss** — для дисбаланса object vs background.
- **Quality Focal Loss** (для centerness).

## Loss = bbox_loss + cls_loss + obj_loss

Веса этих компонент часто критичны. В YOLO они подбираются по grid search.

## Тренировка детектора: typical workflow

1. **Bootstrap dataset.** Разметить 500-2000 картинок.
2. **Стартовый train** с pretrained backbone (COCO).
3. **Eval** на val set: mAP@0.5, mAP@0.5:0.95, per-class mAP.
4. **Анализ ошибок:** false positives, false negatives, misclassifications. Какие классы валятся?
5. **Active learning:** разметить ещё 500 картинок там, где модель уверена меньше всего, или валится.
6. **Retrain.** Goto step 2.

**Итерация важнее, чем выбор архитектуры.** YOLO + 3 итерации > RT-DETR + 1 итерация.

## Аугментации для детекции

Базовый набор (через albumentations):
- HorizontalFlip
- RandomScale (0.5-1.5x)
- ColorJitter
- Mosaic (4 картинки в одну) — критично в YOLO
- MixUp / CutMix

**Не использовать:** rotation > 15° (bbox не повернётся правильно, нужны oriented boxes), heavy crop (может вырезать все объекты).

## Small object detection

Маленькие объекты — самая сложная часть. Что помогает:
- **Высокое разрешение** входа (1024+ вместо 640).
- **FPN (Feature Pyramid Network)** — multi-scale features.
- **Multi-scale training** (разный imgsz во время обучения).
- **Tiling:** разрезаем большую картинку на куски, детектим на каждом, склеиваем.
- **SAHI** library — automated tiling inference.

```python
from sahi.predict import get_sliced_prediction
result = get_sliced_prediction(image, detection_model,
                                slice_height=512, slice_width=512,
                                overlap_height_ratio=0.2)
```

## Annotation

**Tools:**
- **CVAT** — open-source, бесплатно self-host.
- **Label Studio** — open-source, гибкий.
- **Roboflow** — paid, удобный, имеет model-in-the-loop.
- **Supervisely** — paid, для команд.

**Формат:**
- **YOLO format:** `class_id x_center y_center width height` (normalized 0-1), один .txt на картинку.
- **COCO format:** один большой JSON с annotations.
- **Pascal VOC:** XML на картинку.

**Quality:**
- Не делайте annotation сами на 10K — найдёте опечатки и противоречия.
- Используйте guidelines (один документ).
- Inter-annotator agreement: 2 разметчика на 100 случайных картинок, проверяете agreement.

## Active learning

При большом unlabeled pool: используйте модель для выбора uncertain examples.

**Strategies:**
- **Uncertainty:** низкая max confidence → разметить.
- **Diversity:** clustering embeddings, разметить cluster centers.
- **Disagreement:** если ensemble не сходится → разметить.

Экономит 50-80% разметки за тот же gain.

## Production: inference optimization

- **Half precision (FP16/BF16):** 2x speed, no quality loss.
- **TensorRT:** на NVIDIA GPU дополнительно 2-3x.
- **ONNX Runtime:** портабельность.
- **Batching:** если можете батчить — делайте, throughput x2-x5.
- **Resolution:** уменьшение с 640 до 480 даёт 2x speed ценой ~3% mAP.

## Антипаттерны

- **Брать самую большую модель «на всякий случай».** YOLOv11-x на CPU = 5fps. YOLOv11-s = 50fps с -5% mAP.
- **Не делать error analysis.** Просто смотреть mAP — не поймёте, что чинить.
- **Игнорировать class imbalance.** Если 90% объектов — машины, модель будет валить остальные.
- **Mosaic для маленького датасета.** Может ломать обучение, отключите.
- **No augmentation.** Детектор без aug на 1K картинок переобучится.

## Задания

1. Натренируйте YOLOv8 на своём датасете (Roboflow или собственная разметка, 500+ боксов). Цель: mAP@0.5 > 0.7.
2. Сравните YOLOv8-n, -s, -m по mAP и latency. Какой выбрать для CPU? Для GPU?
3. Реализуйте tiling inference для большой картинки (4000×3000) через SAHI. Замерьте gain vs прямой inference.
4. Сделайте error analysis: разделите ошибки на FP/FN/misclassification, постройте confusion matrix per class. Что чинить первым?
5. Реализуйте active learning loop: train → predict on unlabeled → pick top-100 uncertain → label → retrain. На сколько ускоряет labeling?
6. Конвертируйте YOLO в ONNX + TensorRT. Замерьте latency.

## Чек-лист

- [ ] Знаю основные семейства детекторов и их trade-offs.
- [ ] Умею обучить YOLO на своём датасете end-to-end.
- [ ] Понимаю NMS и могу настроить.
- [ ] Делаю error analysis после тренировки.
- [ ] Знаю про small object detection и tiling.
- [ ] Использую active learning для эффективной разметки.

## Дальше

➡️ [08-semantic-segmentation.md](./08-semantic-segmentation.md) — семантическая сегментация.
