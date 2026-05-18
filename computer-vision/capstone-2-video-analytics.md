# Капстон 2. Real-time видео-аналитика с трекингом

End-to-end pipeline: видео-поток → детекция → tracking → бизнес-метрики → dashboard. Цель — система, которая обрабатывает live video и даёт реальную пользу.

## Идеи задач

- **People counter** на входе/выходе магазина с heatmap.
- **Traffic analysis:** подсчёт типов транспорта, speed estimation.
- **Sports analytics:** tracking игроков, статистика движения.
- **Wildlife monitoring:** подсчёт животных у водопоя через timelapse.
- **Industrial safety:** обнаружение людей в danger zones.
- **Queue length** на стойках регистрации.

## Архитектура

```
[Camera/Video] → 
  [Frame decoder (GPU)] → 
  [YOLO detector (TensorRT)] → 
  [ByteTrack tracker] → 
  [Business logic: counter / heatmap / alerts] → 
  [Database (PostgreSQL / TimescaleDB)] → 
  [Streamlit / Grafana dashboard]
                   ↓
          [Alerts: Slack / Telegram]
```

## Структура проекта

```
video-analytics/
├── README.md
├── docker-compose.yml      # PostgreSQL + Grafana + App
├── app/
│   ├── pipeline/
│   │   ├── reader.py       # Async video reading
│   │   ├── detector.py     # YOLO + TensorRT
│   │   ├── tracker.py      # ByteTrack
│   │   ├── analytics.py    # Counters, zones, lines
│   │   └── publisher.py    # DB write, alerts
│   ├── api/
│   │   └── main.py         # FastAPI for control
│   └── ui/
│       └── dashboard.py    # Streamlit dashboard
├── configs/
│   ├── camera1.yaml        # Zones, lines, classes
│   └── tracker.yaml
├── models/
│   └── yolov8m.engine      # TensorRT
├── tests/
└── Dockerfile
```

## Этап 1. Pipeline (3-5 дней)

### Video reading

```python
import cv2
from queue import Queue
from threading import Thread

def video_reader(source: str, frame_queue: Queue):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((frame, time.time()))
```

Для performance: GPU-accelerated decoding через PyAV с CUDA hwaccel.

### Detection + Tracking

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.track(frame, persist=True, tracker='bytetrack.yaml')

for box, track_id, cls in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
    # process tracked object
```

### Business logic: zones и crossing lines

```python
def line_crossed(track_history, line):
    """Check if track trajectory crossed line"""
    if len(track_history) < 2:
        return False
    p1, p2 = track_history[-2], track_history[-1]
    return segments_intersect(p1, p2, line[0], line[1])

class Counter:
    def __init__(self, line):
        self.line = line
        self.crossed_ids = set()
        self.count = 0
    
    def update(self, tracks):
        for tid, history in tracks.items():
            if line_crossed(history, self.line) and tid not in self.crossed_ids:
                self.crossed_ids.add(tid)
                self.count += 1
```

## Этап 2. Database & Dashboard (2-3 дня)

### TimescaleDB schema

```sql
CREATE TABLE events (
    time TIMESTAMPTZ NOT NULL,
    camera_id TEXT,
    event_type TEXT,   -- 'cross_line', 'enter_zone', 'detection'
    track_id INTEGER,
    object_class TEXT,
    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
    confidence REAL,
    metadata JSONB
);
SELECT create_hypertable('events', 'time');
```

### Grafana или Streamlit

- Real-time counts по часам.
- Heatmap движения.
- Avg dwell time в zones.
- Распределение по object classes.

## Этап 3. Optimization (2-3 дня)

### TensorRT

```python
model.export(format='engine', half=True, imgsz=640)
```

Замеряем: 30+ FPS на single A10 для YOLOv8m.

### Multi-stream batching

Для 10+ cameras на одной GPU: batching frames из разных потоков. Triton делает автоматически.

### Frame skipping

Не обязательно обрабатывать каждый кадр. На 30 FPS видео — skip 2-3 frames с tracker interpolation между.

## Этап 4. Alerts & Notifications (1-2 дня)

```python
async def check_alerts(events):
    # Например: > 50 человек в магазине за последний час
    count_last_hour = await db.query("SELECT COUNT(*) FROM events WHERE event_type='cross_line' AND time > NOW() - INTERVAL '1 hour'")
    if count_last_hour > 50:
        await send_telegram_alert(f"High traffic: {count_last_hour}/hour")
```

## Этап 5. Deployment

### Docker Compose

```yaml
services:
  app:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  postgres:
    image: timescale/timescaledb:latest-pg15
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
```

## Чек-лист готовности

- [ ] Pipeline обрабатывает живое видео > 25 FPS.
- [ ] Tracking даёт stable IDs (не switches каждые 5 sec).
- [ ] Counter работает правильно на test video с известным GT.
- [ ] DB schema и события логируются.
- [ ] Dashboard real-time показывает counters/heatmaps.
- [ ] Alerts настроены (Slack/Telegram).
- [ ] Multi-camera support (хотя бы 2).
- [ ] TensorRT для production speed.
- [ ] Docker Compose stack запускается.
- [ ] README с screenshots и demo video.

## Что показать на собесе

1. **Live demo:** запустите на своём ноуте с webcam.
2. **Архитектура** на доске: показать, что понимаете data flow.
3. **Scaling story:** как масштабировать на 100 камер? GPU pooling, async, batching.
4. **Reliability:** что если camera упала? что если detector валится?
5. **Trade-offs:** почему ByteTrack а не DeepSORT, почему TimescaleDB а не Postgres.

## Идея усиления

Добавьте **Re-ID across cameras**: human X на camera 1 = human X на camera 2. Это уже level senior CV engineer.

---

➡️ Следующий: [Капстон 3. Multi-modal RAG над изображениями и документами](./capstone-3-multimodal-rag.md)
