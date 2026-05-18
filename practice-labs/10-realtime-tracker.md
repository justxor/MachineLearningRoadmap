# 🎥 Лаба 10: Real-time трекер людей на видео 🔴

## Цель

Построить real-time трекер, который детектирует людей, присваивает ID и сохраняет их между кадрами. Понять trade-off между FPS и качеством.

## Датасет и видео

- MOT17 / MOT20 (benchmark).
- Pexels, Pixabay — свободные видео с толпой.
- Своя запись (соблюдая приватность).

## Минимальный пайплайн

1. Детектор: YOLOv8n (или nano) — класс person.
2. Трекер: ByteTrack или BoT-SORT.
3. Отрисовка боксов + ID + траектории.
4. Pipeline: VideoCapture → detect → track → draw → write.
5. Измерение FPS.

## Код: ByteTrack pipeline

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model.track(source="video.mp4", classes=[0], tracker="bytetrack.yaml", persist=True)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else []
    # рисуем bbox и id
```

## Метрики

- MOTA, MOTP, IDF1, HOTA.
- ID switches — критически важно для UX.
- FPS на CPU/GPU.
- Latency end-to-end (от кадра до вывода).

## Расширения

- ReID-эмбеддинги (OSNet) для снижения ID switches.
- Counting входящих/выходящих через линию.
- Heatmap посещаемости.
- Экспорт в TensorRT/CoreML для эджа.
- Многопоточный pipeline (декодинг + инференс в разных потоках).

## Критерии приёмки

- [ ] ⊥25 FPS на 720p на среднем GPU.
- [ ] MOTA >50% на MOT17 val.
- [ ] Демо-видео с треками в README.
- [ ] Latency измерена и разобрана по стадиям.
- [ ] Config трекера вынесен в yaml.

## Анти-паттерны

- ❌ Пусть большая модель (YOLOv8x) для real-time.
- ❌ Нет фильтрации классов (детектируем всё подряд).
- ❌ Медленная отрисовка боксов (cv2.putText в цикле без batch).
- ❌ Оценка FPS без учёта декодинга.

---

[← Назад к Practice Labs](./README.md)
