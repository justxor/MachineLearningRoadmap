# Капстон 1. End-to-end детектор объектов на собственном датасете

Полный цикл: от сбора данных до production deployment. Цель — реальный портфолио-проект, который не стыдно показать на собесе.

## Постановка задачи

Выберите узкую полезную задачу:
- Детекция дефектов на промышленной фотографии.
- Подсчёт продуктов на полке (retail).
- Birds species detection.
- Pet breed recognition.
- Дорожные знаки.
- Strawberries ripe vs unripe.
- Защитные средства на стройке (helmet, vest, gloves).

**Цель:** не «ещё один COCO», а **узкая задача с реальным use case**. Это сильнее впечатляет на интервью.

## Структура проекта

```
my-detector/
├── README.md           # Описание, метрики, demo
├── data/
│   ├── raw/            # Сырые картинки (gitignored, DVC)
│   ├── annotations/    # YOLO format labels
│   └── splits.json     # Train/val/test split
├── notebooks/
│   ├── 01_eda.ipynb        # Распределение классов, размеров
│   ├── 02_label_review.ipynb # Sanity check разметки
│   ├── 03_results.ipynb     # Анализ ошибок финальной модели
├── src/
│   ├── data/
│   ├── training/
│   ├── eval/
│   └── serve/
├── configs/
│   └── yolov8m.yaml
├── tests/
├── docker/
├── Makefile
└── pyproject.toml
```

## Этап 1. Сбор данных (1-2 дня)

### Источники
- Google Images (для прототипа, без commercial use).
- Roboflow Universe (готовые open datasets).
- Собственная съёмка.
- Synthetic generation через SD (для редких классов).

### Объём
- Минимум 500 размеченных картинок на класс для прототипа.
- 2000+ для production-quality.

## Этап 2. Разметка (1-3 дня)

### Tool
- **CVAT** (open-source, self-hosted).
- **Label Studio** (open-source, плюс online версия).
- **Roboflow** (online, удобно для small teams).

### Process
1. **Annotation guidelines:** один документ, что считать каждым классом. Иначе разметчики противоречат.
2. **Pilot batch:** 50 картинок, проверка guidelines, agreement check.
3. **Bulk annotation.**
4. **Quality control:** 10% review каждую неделю.

### SAM для acceleration
SAM 2 + box prompts → instance masks для бесплатной generation seg labels.

## Этап 3. EDA и базовая модель (1-2 дня)

### EDA
- Распределение классов (есть ли imbalance?).
- Размеры объектов (small/medium/large по COCO definitions).
- Aspect ratios.
- Сложные случаи: occlusions, low light, small objects.

### Baseline
```bash
# YOLOv11
pip install ultralytics
yolo train data=dataset.yaml model=yolov8m.pt epochs=100 imgsz=640
```

Цель: получить mAP@0.5 > 0.5 в первый день, чтобы знать «нижнюю планку».

## Этап 4. Iteration (1-2 недели)

### Error analysis
- FP/FN/misclassification breakdown.
- Per-class metrics: какой класс валится?
- Hard examples: 20 worst predictions — что общего?

### Improvements
- **More data** для слабых классов (often #1 fix).
- **Tuning hyperparams:** Optuna для key params.
- **Augmentation:** Mosaic, MixUp, RandomCrop с осторожностью.
- **Backbone size:** -m → -l → -x, если хватает GPU.
- **Image size:** 640 → 1024 для small objects.
- **Tile inference** для high-res картинок.

### Target metrics
- mAP@0.5 > 0.7 — хороший baseline.
- mAP@0.5 > 0.85 — production-ready (зависит от домена).
- mAP@[0.5:0.95] > 0.5 — solid.

## Этап 5. Deployment (3-5 дней)

### Export
```bash
yolo export model=runs/train/weights/best.pt format=onnx half=True
```

### FastAPI service

```python
from fastapi import FastAPI, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()
session = ort.InferenceSession('best.onnx', providers=['CUDAExecutionProvider'])

@app.post("/detect")
async def detect(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img_arr = preprocess(img)
    outputs = session.run(None, {'images': img_arr})
    boxes = postprocess(outputs)
    return {"detections": boxes}
```

### Docker
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### Demo
- **Gradio** для interactive web demo.
- Deploy на **Hugging Face Spaces** (бесплатно).

## Этап 6. Monitoring (1-2 дня)

### Metrics
- Prometheus + Grafana для latency, throughput, error rate.
- Confidence distribution histograms.

### Drift detection
- Embedding drift через CLIP на новых картинках.
- Алерт на drift score > threshold.

## Финальный README

Минимум должно быть:
- Проблема и use case.
- Dataset stats (откуда, сколько, классы).
- Approach (model, augmentations, training).
- Results (table с метриками, per-class breakdown, confusion-like analysis).
- Demo link / screenshots.
- How to reproduce (Docker / requirements / commands).
- Limitations (где модель валится).
- Future work.

## Чек-лист готовности

- [ ] Собран dataset 1000+ examples.
- [ ] Annotation guidelines документированы.
- [ ] Train/val/test split с stratification.
- [ ] Baseline trained, metric > 0.5 mAP@0.5.
- [ ] Iteration 3+ раза с improvements.
- [ ] Error analysis написан в notebook.
- [ ] Final mAP@0.5 > 0.7.
- [ ] Exported в ONNX.
- [ ] FastAPI service работает.
- [ ] Docker image билдится и запускается.
- [ ] Live demo на HF Spaces.
- [ ] README с метриками, демо, инструкциями.
- [ ] Тесты для preprocessing и inference.

## Что показать на собесе

1. **Demo** — пусть интервьюер сам попробует.
2. **README** — структурно и понятно.
3. **Error analysis** — это отличает middle от senior.
4. **Trade-offs:** почему YOLO vs RT-DETR, почему 640 vs 1024, почему ResNet vs ConvNeXt.
5. **Production thoughts:** latency, drift, retraining strategy.

## Идея усиления

После базового капстона, добавьте **active learning loop**:
1. Inference на unlabeled pool.
2. Pick top-100 uncertain.
3. Label через CVAT.
4. Retrain.
5. Measure gain.

Это резко повышает уровень проекта.

---

➡️ Следующий: [Капстон 2. Real-time видео-аналитика с трекингом](./capstone-2-video-analytics.md)
