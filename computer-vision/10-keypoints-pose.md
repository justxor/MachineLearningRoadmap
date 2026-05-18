# 10. Keypoints и pose estimation

Найти ключевые точки на объекте: суставы человека, ориентиры лица, точки руки. Применения: AR-фильтры, fitness apps, sign language, sports analytics.

## Типовые задачи

| Задача | Keypoints | Использование |
|--------|-----------|---------------|
| **Human pose** | 17 (COCO) или 33 (MediaPipe) | Sports, fitness, AR |
| **Face landmarks** | 68 или 468 (MediaPipe Face Mesh) | AR-фильтры, FaceID |
| **Hand tracking** | 21 keypoints на руку | Sign language, VR |
| **Whole body** | 133 (COCO-WholeBody) | Full character animation |
| **Animal pose** | Видо-специфично | Biology, veterinary |

## Подходы

### Top-down (детектор + pose model)

1. Детектируем person/object.
2. Для каждого bbox → pose estimator.

Pros: точно на каждом объекте. Cons: медленно при большом количестве объектов.

**Models:** HRNet, ViTPose, RTMPose.

### Bottom-up (heatmap + grouping)

1. Все keypoints всех объектов сразу.
2. Grouping keypoints в скелеты.

Pros: speed не зависит от числа объектов. Cons: качество хуже при оверлапах.

**Models:** OpenPose, HigherHRNet.

### Real-time on-device

**MediaPipe** от Google — реальное std для on-device:
- Pose, Face, Hands.
- Работает на CPU mobile в реальном времени.
- Production-ready, plug-and-play.

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

results = pose.process(image_rgb)
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        x, y, z, visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
```

## Heatmap regression vs coordinate regression

**Heatmap:** модель предсказывает 2D Gaussian heatmap для каждого keypoint, argmax → координаты. Стандарт для качества.

**Coordinate regression:** прямое предсказание (x, y). Проще, иногда хуже.

**Современный:** SimCC (simple coordinate classification) — комбинация: классификация по дискретным координатам, fast + accurate.

## Animal pose / custom keypoints

Pretrained моделей для human — много. Для своего класса (например, dogs):

1. Разметить 500-2000 примеров через CVAT/Label Studio.
2. Fine-tune HRNet/RTMPose на ваших данных.
3. Augmentation специфичная: flip с swap-парных keypoints.

## Метрики

- **OKS** (Object Keypoint Similarity) — см. урок 04.
- **PCK** (Percentage of Correct Keypoints): keypoint правильный, если distance < threshold.
- **MPJPE** (Mean Per-Joint Position Error): средняя ошибка в пикселях или мм для 3D pose.

## 3D pose

2D keypoints → 3D recovery (depth). Тренды:
- **MoGen, HuMoR, OSX**: 3D human pose из single image.
- **VideoPose3D**: 3D pose из 2D последовательности (использует temporal info).
- **NeRF + pose**: 3D mesh fitting (SMPL-X).

## Production: optimization

- **Heatmap output 64×48** вместо 256×192 (4x меньше памяти, малый loss качества).
- **Top-down с детектором YOLO-nano** → RTMPose-tiny: 100+ FPS на CPU mobile.
- **TensorRT** для серверной задачи.

## Антипаттерны

- **Не делать flip с keypoint swap.** Aug бракует данные.
- **Predict (x,y) напрямую** для требовательных задач — heatmap почти всегда лучше.
- **Использовать full body model для face.** Точность лица будет ужасная.
- **Игнорировать visibility flag.** Невидимый keypoint != координаты (0,0).

## Задания

1. Использовать MediaPipe Pose в реальном времени с веб-камерой. Считать reps приседаний.
2. Натренировать собственный keypoint detector на small custom dataset (вашa собака, fish, etc).
3. Сравнить HRNet, ViTPose, RTMPose на COCO val: AP_keypoints и FPS.
4. Реализовать fitness counter (приседания, отжимания) на основе angles между keypoints.
5. Использовать SMPL-X для 3D human mesh из single image.
6. Сравнить top-down и bottom-up при разном количестве людей в кадре (1, 5, 20).

## Чек-лист

- [ ] Понимаю top-down и bottom-up подходы.
- [ ] Умею использовать MediaPipe для on-device.
- [ ] Знаю heatmap vs coordinate regression.
- [ ] Понимаю OKS и PCK.

## Дальше

➡️ [11-tracking.md](./11-tracking.md) — object tracking.
