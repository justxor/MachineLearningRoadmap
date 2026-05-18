# 09. Instance и panoptic сегментация

**Instance:** различает экземпляры одного класса (две машины — две маски).
**Panoptic:** semantic + instance в одной модели.

## Instance сегментация: подходы

### Mask R-CNN (классика, 2017)

Расширение Faster R-CNN: после bbox добавляется маска внутри. До сих пор используется как baseline.

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
predictions = model([image_tensor])
# predictions[0]['masks'], ['boxes'], ['labels'], ['scores']
```

### YOLOv8/v11-seg

YOLO с дополнительной mask head. Быстро, real-time.

```python
from ultralytics import YOLO
model = YOLO('yolov8m-seg.pt')
results = model('image.jpg')
```

### Mask2Former

Универсальный для семантической, instance, panoptic. SOTA quality, но медленнее YOLO.

### SAM + детектор

Современный pattern: детектор находит bbox → SAM сегментирует внутри bbox. Объединяет качество SAM и скорость детектора.

```python
# Pseudo-code
boxes = detector(image)  # YOLO
for box in boxes:
    mask = sam_predictor.predict(box=box)
```

## Panoptic сегментация

Каждому пикселю — либо «вещь» (thing, countable, instance ID), либо «материал» (stuff, uncountable, без instance).

Пример: на улице фотографии «машины» — things (instance 1, 2, ...), «дорога» и «небо» — stuff.

**SOTA:** Mask2Former, OneFormer. Доступны в Hugging Face.

```python
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    'facebook/mask2former-swin-large-cityscapes-panoptic')
```

## Метрики

- **AP_mask:** AP для масок (вместо bbox), как в COCO.
- **PQ (Panoptic Quality):** см. урок 04 про метрики.

## Production scenarios

- **Retail (Amazon Go):** разделение касающихся продуктов на полке.
- **Medical:** отдельные клетки, опухоли.
- **Robotics:** grasping (нужно знать границы объекта).
- **AR:** замена/добавление объектов в кадр.

## Выбор подхода

| Сценарий | Выбор |
|----------|-------|
| Real-time | **YOLOv11-seg** |
| Server, top quality | **Mask2Former** |
| Few examples / no training | **Grounding DINO + SAM** |
| Custom annotation pipeline | **SAM 2 для разметки** |

## Антипаттерны

- Mask R-CNN в 2026, когда есть YOLO-seg или Mask2Former.
- Игнорировать SAM в pipeline — auto-labeling экономит дни работы.
- Не учитывать, что mask AP считается на mask IoU, не bbox.

## Задания

1. Натренируйте YOLOv11-seg на small dataset (LVIS subset или свой). Eval mask AP.
2. Сравните Mask R-CNN, YOLOv11-seg, Mask2Former на одной задаче по quality и speed.
3. Реализуйте pipeline: YOLO детектор + SAM segmentation. Сравните с end-to-end Mask R-CNN.
4. Сделайте panoptic сегментацию через Mask2Former на Cityscapes.
5. Реализуйте instance counting (например, подсчёт людей в кадре через семантику + connected components).
6. Используйте SAM для auto-labeling 200 примеров, исправьте 20% вручную, обучите модель.

## Чек-лист

- [ ] Знаю разницу semantic / instance / panoptic.
- [ ] Умею использовать YOLO-seg и Mask2Former.
- [ ] Понимаю SAM + детектор pattern.
- [ ] Знаю про PQ метрику.

## Дальше

➡️ [10-keypoints-pose.md](./10-keypoints-pose.md) — keypoints и pose estimation.
