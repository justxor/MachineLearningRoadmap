# 18. Edge и mobile inference

CV-инференс на телефоне, NVIDIA Jetson, Raspberry Pi, embedded камерах. Жёсткие constraints по памяти, compute, energy.

## Почему edge

- **Privacy:** данные не уходят в облако (медицина, security).
- **Latency:** real-time без round trip.
- **Bandwidth:** не нужно стримить видео.
- **Offline:** работает без интернета (camera, drone).
- **Cost:** дешевле, чем cloud per inference.

## Hardware ландшафт

| Платформа | Compute | Best for |
|-----------|---------|----------|
| **Apple Silicon (iPhone, iPad, Mac)** | Neural Engine, 15-35 TOPS | High-quality models, AR |
| **Android (Snapdragon, Tensor)** | NPU, 10-50 TOPS | Variable hardware |
| **NVIDIA Jetson Orin** | 70-275 TOPS | Robotics, smart cameras |
| **Raspberry Pi 5** | CPU + maybe Coral USB | Hobby, learning |
| **Coral Edge TPU** | 4 TOPS | Specialized accelerator |
| **Intel Movidius** | VPU | Computer vision focused |

## Optimization pipeline

```
PyTorch model →
ONNX export →
Optimization (constants folding, op fusion) →
Quantization (INT8/INT4) →
Platform-specific compile (CoreML/TFLite/TensorRT) →
On-device benchmark
```

## ONNX export

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx', 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                  opset_version=17)

# Optimize
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

## TensorRT (NVIDIA Jetson, desktop)

NVIDIA-specific compiler. 2-10x speedup vs raw PyTorch на NVIDIA GPU.

```python
# Через torch_tensorrt
import torch_tensorrt
trt_model = torch_tensorrt.compile(model, 
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.half})
```

## CoreML (iOS, macOS)

Через `coremltools`:

```python
import coremltools as ct
mlmodel = ct.convert(model, 
    inputs=[ct.ImageType(shape=(1, 3, 224, 224))],
    convert_to="mlprogram")
mlmodel.save("Model.mlpackage")
```

Apple Neural Engine использует автоматически при правильной конвертации.

## TFLite (Android, mobile)

```python
import tensorflow as tf
# Конвертация через ONNX → TF → TFLite или напрямую

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16
tflite_model = converter.convert()
```

**Hardware acceleration:** NNAPI delegate (Android), GPU delegate, Hexagon (Snapdragon).

## MLX (Apple Silicon)

Native Apple framework, оптимизирован под M-series chips. Сильно лучше PyTorch на Mac для inference.

## Quantization

### INT8 PTQ

Стандарт edge. Используете calibration dataset (~100 примеров) для подбора scales.

```python
import torch.ao.quantization as quant

model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# Calibration
for x in calibration_data:
    model_prepared(x)

model_quantized = quant.convert(model_prepared)
```

**Потеря качества:** обычно 1-3% accuracy.

### INT8 QAT

Quantization-aware training: обучение с симулированным quantization. Качественнее PTQ ценой retrain.

### INT4 / NF4

Для очень маленьких устройств. Качество падает заметнее (5-10%).

## Specialized small models

| Семейство | Use case |
|-----------|----------|
| **MobileNetV3** | Базовая классификация |
| **EfficientNet-B0** | Лучший quality/size trade-off |
| **MobileViT** | ViT для mobile |
| **YOLOv11-n / -s** | Mobile object detection |
| **MobileSAM** | SAM для mobile |
| **TinyCLIP** | CLIP для mobile |

## Benchmarking on-device

Не доверяйте «X ms на GPU». Реальные числа:

```python
import time

# Warmup
for _ in range(10):
    model(input)

# Measure
times = []
for _ in range(100):
    start = time.perf_counter()
    model(input)
    times.append(time.perf_counter() - start)
    
p50 = sorted(times)[50]
p95 = sorted(times)[95]
```

**Метрики:** latency p50/p95, memory peak, energy per inference (для battery).

## Real-world tips

- **Batch=1** обычно на edge. Не оптимизируйте под большие batches.
- **Pre-allocate tensors:** избегайте allocation в hot path.
- **Half precision FP16:** часто бесплатное 2x ускорение на mobile NPU.
- **Skip frames** для real-time video.
- **Adaptive quality:** низкая модель для cold path, тяжёлая по событию.
- **Compile once, run много раз:** не конвертируйте каждый запуск.

## Frameworks comparison

| Framework | iOS | Android | NVIDIA Jetson | Linux/Win edge |
|-----------|-----|---------|----------------|-----------------|
| CoreML | ✅✅ | — | — | — |
| TFLite | ✅ | ✅✅ | ✅ | ✅ |
| ONNX Runtime | ✅ | ✅ | ✅ | ✅✅ |
| TensorRT | — | — | ✅✅ | ✅ (NVIDIA GPU) |
| OpenVINO | — | — | — | ✅ (Intel) |
| MLX | — | — | — | ✅✅ (Apple Silicon) |

## Антипаттерны

- **Тестировать на desktop, ожидать те же FPS на mobile.** Может отличаться в 10-100 раз.
- **FP32 модель на edge.** Зря тратите ресурсы. Минимум FP16.
- **Не использовать platform-native** компилятор (CoreML на iOS, NNAPI на Android).
- **Большая универсальная модель** вместо двух специализированных (быстрая + точная по событию).
- **Не measuring energy.** Battery — критический ресурс mobile.

## Задания

1. Конвертировать YOLOv8-nano в ONNX → TensorRT FP16 на Jetson Nano/Orin. Замерить FPS.
2. Конвертировать ResNet18 в CoreML. Запустить на iPhone, замерить FPS и температуру.
3. Сравнить INT8 PTQ vs QAT для классификации на MobileNetV3. Разница в accuracy.
4. Использовать MLX для inference Stable Diffusion на Mac Mini M2. Сколько секунд на картинку?
5. Реализовать «two-tower» edge pipeline: маленькая модель всегда, большая по trigger.
6. Замерить energy consumption MobileNet vs EfficientNet на телефоне (battery historian).

## Чек-лист

- [ ] Знаю основные edge platforms и их constraints.
- [ ] Умею экспортировать в ONNX/CoreML/TFLite.
- [ ] Понимаю INT8 PTQ и могу применить.
- [ ] Знаю, какие модели подходят для mobile (MobileNet, EfficientNet, YOLO-nano).
- [ ] Умею benchmarking on-device (latency, memory, energy).

## Дальше

➡️ [19-production-deploy.md](./19-production-deploy.md) — деплой CV-моделей в прод.
