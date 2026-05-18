# 17. Видео-аналитика

Видео = последовательность кадров + временной context. Применения: surveillance, retail analytics, sports, autonomous driving, content moderation.

## Подходы к видео-задачам

### Frame-by-frame

Обрабатываем каждый кадр независимо моделью для картинок (детекция, сегментация). + tracker для consistency.

**Когда:** real-time, простые задачи без temporal context.

**Pros:** дешёво, легко scale. **Cons:** не учитывает движение между кадрами.

### Sliding window

Берём окно из N кадров (8, 16, 32) → temporal model.

**Models:** I3D, X3D, SlowFast (3D CNN), TimeSformer, MViT (трансформеры).

```python
import torch
from torchvision.models.video import r3d_18, mvit_v2_s
model = mvit_v2_s(weights='KINETICS400_V1')
```

### Streaming / online

Real-time processing с накоплением state (RNN, transformer with cache). Используется в self-driving.

## Action recognition

Классификация коротких клипов (Kinetics-400, что человек делает).

**Стандарты:**
- **Video Swin Transformer**
- **MViT-v2**
- **VideoMAE** (self-supervised pretrain)

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# inputs: 16 frames at 224x224
```

## Temporal action detection / localization

Не «что происходит», а «когда что происходит» в длинном видео.

Подходы: anchor-based (как detection), DETR-like, two-stage.

**Models:** ActionFormer, TallFormer, GTAD.

## Video object tracking

См. урок 11. ByteTrack + frame-by-frame детектор — стандарт.

## Анализ видео через VLM

С 2024 года VLM (Qwen2-VL, Gemini, GPT-4o) умеют принимать видео на вход:

```python
# Qwen2-VL может принять frames из видео
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "video.mp4", "max_pixels": 360*420, "fps": 1.0},
        {"type": "text", "text": "Describe what happens in this video."},
    ],
}]
```

**Применения:**
- Содержательное описание видео.
- Q&A по видео.
- Highlight extraction.
- Content moderation.

**Cons:** медленно, дорого. На длинных видео — нужна разбивка.

## Real-time pipeline

```
Camera → Frame queue → 
[Detector + Tracker] → 
Business logic (counting, alerts) → 
[Database + Dashboard]
```

### Frame skipping

Skipping 2-3 frames между detections даёт 2-3x speedup без сильной потери качества (tracker заполняет интерполяцией).

### Multi-stream batching

10 потоков камер → batch их кадры через одну GPU. Throughput ×5-×10 vs обработка каждого потока отдельно.

### Hardware-accelerated decode

NVIDIA NVENC/NVDEC: декодирование видео на GPU. Освобождает CPU. Через PyAV или DALI.

## Применения

### Подсчёт людей / транспорта

Стандарт: YOLO + ByteTrack + crossing line. См. урок 11.

### Анализ траекторий

Recording всех treclka в видео → heatmaps, dwell time, popular paths. Используется в retail (как покупатель ходит).

### Anomaly detection

Detection unusual events. Подходы:
- **Сlassification:** обучаем «normal vs abnormal».
- **Reconstruction-based:** autoencoder восстанавливает нормальное, ошибка → аномалия.
- **One-class SVM / Isolation Forest** на features.

Часто работает плохо на real-world data из-за разнообразия normal events.

### Activity recognition в спорте

«Игрок забил гол», «спортсмен сделал прыжок». Pretrained pose models + classifier на keypoint sequences.

### Self-driving CV

Multi-task: depth + segmentation + lane detection + 3D object detection. **Modern stack:** BEV (Bird's Eye View) представление, fusion камер.

## Storage

Видео-данные ОГРОМНЫЕ. Strategies:
- Сохранять только **events** (когда что-то происходит), не весь поток.
- Cloud storage (S3) с lifecycle policies.
- Downsample resolution и FPS для archive.
- Сохранять **inferred events** в БД, raw — на 7-30 дней.

## Cost optimization

Real-time video с ML — дорого. На 1000 камер:
- Skip frames агрессивно (4-5).
- Маленькая модель для триггера, большая по событию.
- Edge processing где возможно (NVIDIA Jetson).

## Антипаттерны

- **Полные frames для длинных видео.** Разбивайте, обрабатывайте chunks.
- **Не использовать tracker.** Каждый frame independent → ID switches каждую секунду.
- **Игнорировать FPS variability.** Real video часто 23.976/29.97/59.94 fps. Скорость инференса ≠ скорость видео.
- **Single threaded video reading.** Декодирование может быть bottleneck. Используйте `decord` или multi-threaded `PyAV`.

## Задания

1. Реализовать people counter на real-time video stream (webcam). Считать unique people проходящих через линию.
2. Использовать VideoMAE для action classification на UCF-101 subset.
3. Реализовать highlight extraction из 30-минутного видео через VLM.
4. Замерить latency и throughput pipeline для 10 одновременных video streams через batching.
5. Реализовать anomaly detection: autoencoder learns normal scenes, alert на reconstruction error.
6. Использовать Qwen2-VL для извлечения key events из футбольного матча.

## Чек-лист

- [ ] Знаю подходы: frame-by-frame, sliding window, streaming.
- [ ] Умею использовать MViT/VideoMAE для action classification.
- [ ] Понимаю VLM для video understanding.
- [ ] Знаю оптимизации: skip frames, batching, hardware decoding.

## Дальше

➡️ [18-edge-mobile.md](./18-edge-mobile.md) — edge и mobile inference.
