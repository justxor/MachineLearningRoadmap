# 19. Деплой CV-моделей в прод

Сервер для CV — Triton, FastAPI, BentoML. Batching, GPU pooling, auto-scaling. Real-world считай не «accuracy», а «accuracy + latency + cost + reliability».

## FastAPI: простой деплой

```python
from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io

app = FastAPI()
model = torch.jit.load('model.pt').cuda().eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(tensor)
    return {"class": logits.argmax(1).item(), "confidence": logits.softmax(1).max().item()}
```

**Когда:** прототип, низкий трафик. **Не для:** batch optimization, GPU sharing.

## Triton Inference Server (стандарт production)

NVIDIA Triton — лучший сервер для multi-model GPU inference. Поддерживает TensorRT, ONNX, PyTorch, TF.

**Возможности:**
- **Dynamic batching:** автоматически группирует requests.
- **Multi-model:** одновременно работают разные модели на одной GPU.
- **Model versioning:** rollover между версиями.
- **Performance metrics:** built-in Prometheus.

**Структура:**
```
model_repository/
└── my_model/
    ├── config.pbtxt
    └── 1/
        └── model.plan  # TensorRT
```

```
# config.pbtxt
name: "my_model"
platform: "tensorrt_plan"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}
input [{name: "input", data_type: TYPE_FP16, dims: [3, 224, 224]}]
output [{name: "output", data_type: TYPE_FP16, dims: [1000]}]
```

Запуск: `docker run --gpus all -p 8000:8000 nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models`

## BentoML

Python-friendly альтернатива Triton. Не такой быстрый, но проще для R&D и middle-scale.

```python
import bentoml
runner = bentoml.pytorch.get('model:latest').to_runner()
svc = bentoml.Service('cv_svc', runners=[runner])

@svc.api(input=Image(), output=JSON())
def predict(img):
    return runner.run(preprocess(img))
```

## TorchServe

PyTorch-native. Хорош, если уже в PyTorch stack. Менее популярный чем Triton.

## Batching strategies

### Dynamic batching

Сервер собирает запросы в окне (5-50ms) и обрабатывает batch. Throughput x2-x10, latency немного больше.

### Sequence batching

Для stateful models (RNN, KV-cache). Triton поддерживает.

### Continuous batching

Token-level batching (для LLM). Не релевантно для CV.

## GPU sharing

### MIG (Multi-Instance GPU) на A100/H100

Разделение физической GPU на изолированные slices. Каждой модели — свой slice.

### MPS (Multi-Process Service)

Несколько процессов на одной GPU без MIG. Менее изолировано, больше throughput на маленьких моделях.

### Triton's instance_group

Несколько копий модели в памяти GPU. Запросы балансируются между ними.

## Pre/post-processing

Часто preprocessing занимает 30-50% времени! Optimization:

- **GPU preprocessing** через DALI или torchvision.transforms.v2 на GPU.
- **Triton ensemble:** preprocessing как отдельная модель в pipeline.
- **Cython/Numba** для CPU-bound transforms.

## API design

### Sync vs Async

- **Sync (REST):** простой, для real-time (<100ms).
- **Async (queue + webhook):** для тяжёлых задач (long videos, batch).
- **WebSocket:** для streaming (live video).

### Request format

- **Multipart/form-data:** для file upload, simple.
- **JSON + base64:** universal, но 33% overhead.
- **gRPC:** быстро, бинарный, для service-to-service.

### Response

- **Synchronous:** результат сразу.
- **Polling:** клиент опрашивает job_id.
- **Webhook:** push к клиенту по готовности.

## Auto-scaling

### Horizontal Pod Autoscaler (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: {type: Utilization, averageUtilization: 70}
```

**Для GPU:** scale по GPU utilization (custom metric через Prometheus + DCGM).

### Spot/preemptible nodes

Для batch jobs — спот-инстансы в 3-5x дешевле. Не для real-time.

## Caching

- **Result cache:** одинаковый запрос → одинаковый результат. Redis с TTL.
- **Embedding cache:** если pipeline считает embeddings — кэшируйте.
- **CDN:** для готовых картинок (thumbnails).

## Cost analysis

A100 на cloud (2026):
- AWS p4d.24xlarge (8 A100): ~$30/hour.
- Spot: ~$10-15/hour.

Self-hosted on-prem окупается при стабильной 24/7 утилизации > 30%.

## Reliability

### Health checks

```python
@app.get("/health")
def health():
    # Прогон inference на dummy input
    try:
        model(torch.zeros(1, 3, 224, 224).cuda())
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(503, str(e))
```

### Circuit breaker

Если downstream сервис падает — не вешать всю систему. Через библиотеки tenacity, py-breaker.

### Graceful degradation

Большая модель упала → fallback на меньшую. Лучше плохой результат, чем 503.

## Антипаттерны

- **Flask без worker'ов для CV.** Однопоточно — упрётесь в CPU.
- **FastAPI без batching на GPU.** Underutilization.
- **Не warmup перед первым real request.** Cold start = 5-30 sec.
- **Шифрование больших изображений в JSON через base64.** Огромный overhead, используйте multipart или S3 presigned URLs.
- **Не considering preprocessing cost.** Часто половина latency — это resize+normalize.

## Задания

1. Развернуть YOLOv8 в Triton с TensorRT. Замерить throughput vs FastAPI.
2. Реализовать dynamic batching в FastAPI вручную через asyncio queue.
3. Сравнить sync REST API vs gRPC для CV inference на 1000 RPS.
4. Запустить две модели разного размера на одной GPU через Triton, обеспечить fair queuing.
5. Реализовать caching результатов через Redis с image hash в качестве ключа.
6. Загрузить в K8s, настроить HPA по GPU utilization.

## Чек-лист

- [ ] Знаю Triton/BentoML/TorchServe и могу выбрать.
- [ ] Понимаю dynamic batching и его trade-offs.
- [ ] Умею оптимизировать preprocessing.
- [ ] Знаю sync vs async API patterns.
- [ ] Понимаю GPU sharing strategies.

## Дальше

➡️ [20-monitoring-mlops.md](./20-monitoring-mlops.md) — мониторинг и MLOps для CV.
