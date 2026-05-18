# 08. Семантическая сегментация

Каждому пикселю — класс. Не различает экземпляры (две машины — одна маска).

## Архитектуры: краткая история

- **FCN** (2014): первая end-to-end сетка для сегментации. Учебная классика.
- **U-Net** (2015): encoder + decoder + skip connections. До сих пор baseline для медицины.
- **DeepLab v3+**: atrous convolutions + ASPP. Хорош на complex scenes.
- **Mask2Former** (2022): универсальный трансформер для seg/instance/panoptic. SOTA.
- **SAM 2** (2024): foundation model для сегментации. Promptable, zero-shot.

## U-Net на пальцах

```
Encoder: 256→128→64→32 (downsample, growing channels)
              ↓        ↓        ↓
Decoder: 32→64→128→256 (upsample + skip from encoder)
```

Skip connections передают высокоразрешённые детали в decoder. Без них сегментация была бы blurry.

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=NUM_CLASSES,
    activation=None  # logits, softmax/sigmoid в loss
)
```

`segmentation_models_pytorch` (smp) — стандарт для быстрых experiments с U-Net, FPN, DeepLab, PSPNet, MA-Net и др.

## Loss-функции

| Loss | Когда |
|------|-------|
| **Cross-entropy** | Default, multi-class |
| **BCE + Dice** | Бинарная или multi-label сегментация (медицина) |
| **Focal Loss** | Сильный class imbalance |
| **Tversky / Lovasz** | Дисбаланс, оптимизирует IoU напрямую |
| **Boundary Loss** | Когда важны точные границы |

```python
loss = 0.5 * nn.CrossEntropyLoss()(logits, targets) + 0.5 * smp.losses.DiceLoss(mode='multiclass')(logits, targets)
```

Combo CE + Dice — стандарт для несбалансированной сегментации.

## SAM 2: foundation model

**Segment Anything Model 2** от Meta — promptable сегментация:
- Click → mask of object under cursor.
- Box → mask of object in box.
- Text → mask via grounded SAM.

```python
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, _, _ = predictor.predict(point_coords=np.array([[500, 300]]),
                                  point_labels=np.array([1]))
```

**Применения:**
- **Auto-labeling:** SAM генерирует маски, human review → быстрая разметка.
- **Interactive editing:** Photoshop-style выделение.
- **Zero-shot segmentation:** без обучения, для general objects.

## Mask2Former

Универсальная архитектура для семантической, instance и panoptic сегментации. Трансформер с masked attention.

```python
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
```

В 2026 — SOTA на Cityscapes, ADE20K и многих других benchmarks.

## Размер маски: full-res vs downsample

Полный resolution → много памяти. Стандартные решения:

- **Сегментация на 1/4 разрешении → upsampling.** OK для большинства задач.
- **Sliding window inference** на больших картинках.
- **High-res guidance:** маленькая модель на full-res делает refinement маски от большой модели на low-res.

## Класс «фон»

Часто включают «фон» как ещё один класс. Подвохи:
- Фон доминирует по площади → дисбаланс.
- Класс-веса в loss или Dice по другим классам помогают.
- В eval mIoU обычно усредняем **без** фона.

## Семантическая сегментация в production

**Сценарии:**
- **Background removal** (Zoom-style виртуальные фоны).
- **Парсинг документов** (где текст, где таблица).
- **Медицина** (опухоли на МРТ).
- **Satellite imagery** (land cover).
- **Autonomous driving** (road, sidewalk, person).

**Latency optimization:**
- Меньшая модель + смотри только нужные классы.
- TensorRT с INT8.
- Resolution adaptation (full-res только для critical objects).

## Антипаттерны

- **Только CE loss на дисбалансе.** Модель предсказывает только majority class.
- **Bilinear upsampling масок.** Создаёт «промежуточные» классы. Используйте nearest.
- **Полный resolution сразу.** Память кончится. Train на 512×512, потом fine-tune на 1024×1024.
- **Не визуализировать предсказания во время обучения.** Метрики могут идти вверх, но визуально — мусор.

## Задания

1. Натренируйте U-Net на Oxford Pets или Cityscapes (subset). mIoU > 0.6.
2. Сравните **U-Net + ResNet50** vs **DeepLab v3+** vs **Mask2Former** на одной задаче.
3. Используйте SAM 2 для auto-labeling: на 100 картинках получите маски за минуты, исправьте вручную, обучите модель.
4. Реализуйте combo loss CE + Dice. На каком соотношении лучшее качество?
5. Сделайте sliding window inference на картинке 4000×4000.
6. Конвертируйте U-Net в ONNX, запустите inference на CPU, замерьте latency.

## Чек-лист

- [ ] Знаю архитектуры: U-Net, DeepLab, Mask2Former.
- [ ] Умею использовать `segmentation_models_pytorch`.
- [ ] Понимаю Dice/Focal/Tversky loss.
- [ ] Использую SAM 2 для auto-labeling.
- [ ] Применяю sliding window для больших картинок.

## Дальше

➡️ [09-instance-segmentation.md](./09-instance-segmentation.md) — instance и panoptic сегментация.
