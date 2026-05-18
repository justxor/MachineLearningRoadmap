# 11. Object tracking

Tracking = соединение детекций между кадрами через постоянные IDs. Каждому объекту назначается track_id, сохраняемый, пока объект виден.

## SOT vs MOT

- **SOT (Single Object Tracking):** один объект, инициализация bbox-ом в первом кадре. Применения: AR, военная техника, sports player tracking.
- **MOT (Multi Object Tracking):** все объекты одного/нескольких классов. Применения: traffic analysis, retail analytics, sports.

## Tracking-by-detection paradigm (стандарт MOT)

1. На каждом кадре — детектор (YOLO/RT-DETR).
2. Tracker связывает детекции с существующими треками.

**Связывание (association):** Hungarian algorithm + cost matrix (IoU, motion, appearance features).

## Алгоритмы

### SORT (2016)

Kalman filter для предсказания + Hungarian assignment по IoU. Простой, быстрый, baseline.

### DeepSORT (2017)

SORT + Re-ID embedding (CNN feature) для appearance matching. Меньше ID switches при оверлапах.

### ByteTrack (2022)

Использует **low-confidence detections** на втором этапе ассоциации — резко снижает FN при partial occlusion. **Стандарт в 2026.**

```python
# ultralytics integration
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model.track(source='video.mp4', tracker='bytetrack.yaml')
for r in results:
    for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
        # process tracked object
```

### BoT-SORT, StrongSORT, OC-SORT

Современные SOTA, итерации над ByteTrack. Реальная разница для большинства задач — небольшая, BoT-SORT часто чуть лучше.

### MOTRv2, TrackFormer

End-to-end transformer-based tracking. Многообещают, но в проде ещё не стандарт.

## Метрики

- **MOTA** (Multi-Object Tracking Accuracy): учитывает FP, FN, ID switches.
- **MOTP**: precision локализации.
- **IDF1**: F1 на уровне identities. Главная для long-term tracking.
- **HOTA**: балансирует detection и association. **Современный стандарт.**

## Production challenges

### ID switches

Когда два объекта пересекаются, треки могут поменяться. Лечения:
- Re-ID embedding (DeepSORT, BoT-SORT).
- Motion model (Kalman filter).
- Track history (Iou over time, не только last bbox).

### Occlusion handling

Объект исчез на 30 кадров → нужно «помнить» его, чтобы re-associate. Параметры:
- `track_buffer`: сколько кадров держать инвидимый track.
- `new_track_thresh`: confidence для нового track.

### Real-time pipeline

Detection обычно — узкое место. Tracking сам по себе очень быстрый.

**Optimization:**
- Skip frames: детектить каждый 2-й или 3-й кадр, между ними — motion prediction tracker'а.
- Smaller model для детекции, если допустимо.
- Batch inference нескольких видеопотоков одновременно.

## Применения

| Сценарий | Tracker | Особенности |
|----------|---------|-------------|
| **Подсчёт людей на входе** | ByteTrack + crossing line | Зона + направление |
| **Анализ траекторий покупателей** | ByteTrack + Re-ID | Long-term, heatmaps |
| **Traffic analysis** | ByteTrack | Speed estimation через homography |
| **Sports player tracking** | StrongSORT + Re-ID | Замена игроков, jersey numbers |
| **Action recognition** | Tracking + temporal model | I3D, X3D, MViT |

## Counting и crossing line detection

Часто tracker'а достаточно для бизнес-задачи:

```python
# Pseudo-code
line = ((x1, y1), (x2, y2))  # виртуальная линия
crossed = set()

for frame in video:
    tracks = tracker.update(detections)
    for track in tracks:
        if line_crossed(track.history, line) and track.id not in crossed:
            crossed.add(track.id)
            count += 1
```

## Связь с Re-ID

Person Re-Identification — отдельная задача: дано фото человека, найти его на других камерах / в другое время. Используется в multi-camera tracking.

**Models:** OSNet, BPB-Net. Доступны в torchreid.

## Антипаттерны

- **Игнорировать low-confidence detections.** SORT их выбрасывает, ByteTrack умеет ассоциировать → меньше fragmentation.
- **track_buffer = 30** для short videos. Лишние ghost-треки. Подбирайте под scenarios.
- **Не учитывать camera motion.** Если камера движется (drone) — простой IoU tracking ломается. Нужна камера-стабилизация или global motion compensation.
- **Confound MOTA и IDF1.** MOTA можно «накрутить» за счёт детекции, IDF1 точнее показывает quality tracking.

## Задания

1. Запустить YOLOv8 + ByteTrack на любом видео из RoadCAM. Замерить FPS.
2. Реализовать counter людей через crossing line.
3. Сравнить ByteTrack vs BoT-SORT на стандартном MOT17 benchmark. Разница в HOTA.
4. Реализовать multi-camera Re-ID: matched IDs между двумя камерами.
5. Замерить деградацию tracking при skip-frame=2,3,5. Какая граница приемлемого?
6. Построить heatmap движения людей за час видео из магазина.

## Чек-лист

- [ ] Понимаю tracking-by-detection paradigm.
- [ ] Знаю SORT/DeepSORT/ByteTrack.
- [ ] Понимаю MOTA, IDF1, HOTA.
- [ ] Умею интегрировать ByteTrack в pipeline.
- [ ] Знаю counting через crossing lines.

## Дальше

➡️ [12-vision-transformers.md](./12-vision-transformers.md) — Vision Transformers.
