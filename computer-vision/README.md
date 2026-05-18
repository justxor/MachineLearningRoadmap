# 👁️ Computer Vision Engineering — практический курс

Шестой модульный курс этого репозитория. От базовых операций над картинками до production-систем для детекции, сегментации, OCR, трекинга и мультимодальных моделей. Без воды, без академических доказательств — только то, что реально нужно CV-инженеру в 2026.

## 🎯 Для кого этот курс

- Знаете Python и базовый DL (PyTorch, обучение нейросетей).
- Хотите специализироваться в CV или укрепить CV-компетенции в проекте.
- Нужны прикладные навыки: not «обучил CNN на MNIST», а «задеплоил детектор в прод с p95 < 50ms».

**Если нет базы:** сначала пройдите [math-for-ml](../math-for-ml/README.md) и [neural-networks](../neural-networks/README.md). Этот курс предполагает, что вы понимаете, что такое свёртка и тренировочный цикл.

## 📚 Структура курса

**21 урок, 4 блока + 3 капстона.**

### Блок 1. Фундамент CV

- [01. Цифровые изображения и базовые операции](./01-image-basics.md) — пиксели, форматы, цветовые пространства, OpenCV.
- [02. Классические алгоритмы CV](./02-classical-cv.md) — фильтры, edges, features (SIFT, ORB), гомография.
- [03. Аугментации и data pipelines](./03-augmentations.md) — albumentations, batch processing, кастомные трансформации.
- [04. Метрики CV](./04-metrics.md) — accuracy, IoU, mAP, panoptic quality.
- [05. Backbones и transfer learning](./05-backbones.md) — ResNet, EfficientNet, ConvNeXt, fine-tuning стратегии.

### Блок 2. Основные задачи

- [06. Классификация изображений](./06-classification.md) — от baseline до SOTA, multi-label, imbalance.
- [07. Детекция объектов](./07-detection.md) — YOLO, DETR, RT-DETR, anchor-based vs anchor-free.
- [08. Семантическая сегментация](./08-semantic-segmentation.md) — U-Net, DeepLab, Mask2Former.
- [09. Instance и panoptic сегментация](./09-instance-segmentation.md) — Mask R-CNN, SAM 2.
- [10. Обнаружение ключевых точек и pose estimation](./10-keypoints-pose.md) — HRNet, MediaPipe.
- [11. Object tracking](./11-tracking.md) — SORT, ByteTrack, multi-object tracking.

### Блок 3. Vision Transformers и мультимодальность

- [12. Vision Transformers](./12-vision-transformers.md) — ViT, Swin, DINO, MAE.
- [13. CLIP и мультимодальные модели](./13-clip-multimodal.md) — zero-shot classification, image-text retrieval.
- [14. VLM: vision-language models](./14-vlm.md) — LLaVA, Qwen-VL, Florence-2, OCR-задачи.
- [15. Генеративные модели для CV](./15-generative-cv.md) — Stable Diffusion, ControlNet, inpainting, super-resolution.

### Блок 4. Production

- [16. OCR и распознавание документов](./16-ocr.md) — Tesseract, PaddleOCR, donut, LayoutLM.
- [17. Видео-аналитика](./17-video-analytics.md) — temporal models, action recognition, real-time pipelines.
- [18. Edge и mobile inference](./18-edge-mobile.md) — ONNX, CoreML, TFLite, NVIDIA Jetson, квантование.
- [19. Деплой CV-моделей в прод](./19-production-deploy.md) — Triton, FastAPI, batching, GPU pooling.
- [20. Мониторинг и MLOps для CV](./20-monitoring-mlops.md) — drift на картинках, retraining, A/B тесты визуальных моделей.
- [21. Безопасность и privacy в CV](./21-safety-privacy.md) — adversarial attacks, deepfake detection, face anonymization, лицензирование данных.

### 🎯 Капстон-проекты

- [Капстон 1. End-to-end детектор объектов на собственном датасете](./capstone-1-detector.md) — от сбора и разметки до деплоя.
- [Капстон 2. Real-time видео-аналитика с трекингом](./capstone-2-video-analytics.md) — детекция + tracking + count + dashboard.
- [Капстон 3. Multi-modal RAG над изображениями и документами](./capstone-3-multimodal-rag.md) — VLM + OCR + retrieval + LLM.

## 🛠️ Стек курса

**Базовый:** Python 3.12+, PyTorch 2.x, OpenCV, Pillow, albumentations.

**Модели:** torchvision, timm (модели CNN/ViT), ultralytics (YOLO), Hugging Face transformers.

**Деплой:** ONNX, ONNX Runtime, NVIDIA Triton, FastAPI, Docker.

**Eval:** torchmetrics, COCO API, FiftyOne.

**Аннотации:** CVAT, Label Studio, Roboflow.

## ⏱️ Сколько займёт

- **Интенсив (25+ ч/нед):** 6-8 недель.
- **С работой (10-12 ч/нед):** 4-6 месяцев.
- **Спокойный темп (5-7 ч/нед):** 9-12 месяцев.

**Правило:** одна тема в неделю + один практический артефакт. Не торопитесь — лучше глубоко по 21 теме, чем поверхностно по 50.

## 🎯 Что вы получите на выходе

- **Понимание изнутри:** от пикселей до Vision Transformer — без чёрных ящиков.
- **6 готовых архитектур:** классификатор, детектор, сегментатор, трекер, VLM-приложение, OCR-pipeline.
- **3 production-ready капстона** для портфолио.
- **Прокачанный MLOps для CV:** drift detection, A/B-тесты визуальных моделей, edge-деплой.
- **Понимание trade-offs:** YOLO vs DETR, full fine-tune vs transfer learning, on-device vs cloud.
- **Знание ниши:** OCR, видео-аналитика, медицинский CV, autonomous systems — куда расти после base.

## 📊 Карьерный контекст

**Computer Vision Engineer** — одна из самых стабильно востребованных ML-ролей. Меньше хайпа, чем у LLM, но больше реальных задач в индустрии: e-commerce (визуальный поиск, photo tagging), retail (видео-аналитика магазинов), security (face recognition, fraud), медицина (медснимки), agritech, autonomous vehicles, AR/VR.

**Грейды и зарплаты (Россия / международный рынок, 2026):**

| Уровень | RU (нетто/мес) | International (gross/год) |
|---------|----------------|---------------------------|
| Junior CV | 150-250K ₽ | $70-100K |
| Middle CV | 300-500K ₽ | $120-180K |
| Senior CV | 500-900K ₽ | $180-300K |
| Staff/Principal CV | 900K+ ₽ | $300-500K+ |

**Hot subdomains 2026:** мультимодальные модели (VLM), видео в реальном времени (action recognition, tracking), edge AI, foundation models для CV (DINO, SAM), generative CV (стилизация, super-res, inpainting), 3D vision и NeRF.

## 🔗 Связь с другими курсами репозитория

- **Перед этим курсом:** [math-for-ml](../math-for-ml/README.md) — нужна линейная алгебра и понимание свёрток.
- **Параллельно:** [neural-networks](../neural-networks/README.md) — углубление в архитектуры (ViT, MAE).
- **После:** [llm-engineering](../llm-engineering/README.md) для VLM-проектов; [data-science](../data-science/README.md) для product-side задач (A/B на CV-фичах).
- **Усиление:** [claude-code](../claude-code/README.md) для ускорения разработки CV-pipeline.

## 🚫 Чего в этом курсе НЕТ (намеренно)

- **Doomy теория CNN.** Без 50 страниц доказательств про эквивариантность к сдвигу. Достаточно интуиции + рабочего кода.
- **Устаревшие модели.** AlexNet и VGG упомянуты исторически, но не учим обучать. Стандарт — ResNet/EfficientNet/ConvNeXt + ViT/Swin.
- **Hype без практики.** Нет «10 cool tricks с CLIP» — есть «как засунуть CLIP в прод для поиска по картинкам».
- **Generative как самоцель.** Есть применение (super-res, inpainting), но не «как нарисовать арт».

## 📚 Дополнительные ресурсы

**Курсы:**
- [Stanford CS231n](http://cs231n.stanford.edu/) — классика.
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/) — современный, бесплатный.
- [fast.ai Practical Deep Learning](https://course.fast.ai/) — practical-first.

**Книги:**
- *Computer Vision: Algorithms and Applications* — Richard Szeliski (бесплатно онлайн).
- *Deep Learning for Computer Vision* — Rajalingappaa Shanmugamani.
- *Modern Computer Vision with PyTorch* — V Kishore Ayyadevara.

**Конференции:** CVPR, ICCV, ECCV — главные CV-конференции, обзоры на YouTube бесплатно.

**Блоги и Telegram:**
- [Lil'Log](https://lilianweng.github.io/) — глубокие технические разборы.
- [Roboflow Blog](https://blog.roboflow.com/) — практические туториалы.
- [@ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data) — релизы CV-моделей.

## ✅ Чек-лист готовности к курсу

- [ ] Python на уровне «свободно пишу скрипты и классы».
- [ ] PyTorch: понимаю Dataset, DataLoader, тренировочный цикл.
- [ ] Знаю, что такое CNN на интуитивном уровне (свёртка, pooling).
- [ ] Есть GPU (минимум RTX 4060 локально) или готов работать в Colab/Kaggle.
- [ ] Git, Docker базово.

**Если что-то не отмечено:** возвращайтесь к базовым курсам перед стартом.

---

▶︎ Начать: [01-image-basics.md](./01-image-basics.md)
