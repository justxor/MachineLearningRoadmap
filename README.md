# 🤖 Machine Learning Roadmap: от базы до гуру вайбкодинга

> **Карта обучения машинному обучению (Machine Learning, Deep Learning, LLM, Generative AI, MLOps)** — от первого `import numpy` до уровня инженера, который понимает, **как ИИ работает внутри**, и может писать прод‑системы, а не только дёргать API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Roadmap](https://img.shields.io/badge/Roadmap-2025--2026-blue.svg)](#)
[![Made for](https://img.shields.io/badge/Made%20for-RU%20ML%20community-red.svg)](#)

**Ключевые слова:** машинное обучение, глубокое обучение, ML roadmap, LLM, нейросети, PyTorch, transformers, RAG, AI агенты, MLOps, fine‑tuning, prompt engineering, vibe coding, generative AI, Hugging Face, дата‑сайентист, ML инженер, как стать ML инженером, обучение машинному обучению с нуля.

---

## 📺 Полезные Telegram‑каналы (читать каждый день)

- **[@csharp_ci](https://t.me/csharp_ci)** — общеайтишный канал, шорты и дайджесты.
- **[@csharp_1001_notes](https://t.me/csharp_1001_notes)** — заметки по C# и общим инженерным темам, помогает держать тонус.
- **[@ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data)** — **главный русскоязычный канал по ML/AI/Big Data**. Свежие статьи, релизы моделей, разборы.
- **[📁 Большая папка ML/AI каналов](https://t.me/addlist/u15AMycxRMowZmRi)** — кураторская подборка лучших каналов по машинному обучению, нейросетям, LLM и MLOps. Подписаться оптом.

> 💡 Совет: не подписывайтесь на 200 каналов. Возьмите 5–7, читайте каждый день 15 минут — этого хватит, чтобы быть в курсе индустрии.

---

## 🎯 TL;DR

Этот roadmap — не «список курсов на год». Это **карта местности**, по которой вы прокладываете свой маршрут. Цель — не «пройти курс», а **уметь делать**: тренировать модели, поднимать инференс, строить RAG, дообучать LLM, мониторить прод, понимать статьи и читать чужой код.

Ориентир по времени при ~10–15 часов в неделю:

- **0–3 мес:** Python + математика + классический ML → первые модели на табличных данных.
- **3–6 мес:** Deep Learning, CV, NLP → собственные нейросети на PyTorch.
- **6–12 мес:** LLM, transformers, RAG, fine‑tuning, AI‑агенты → прикладные проекты.
- **12+ мес:** MLOps, прод, scaling, специализация → джуниор → мидл уровень в реальной работе.

---

## 🧭 5 правил выживания

1. **Кода больше, чем теории.** Каждая тема закрывается своим артефактом: ноутбуком, репозиторием, демо.
2. **Не учите всё сразу.** Один трек за раз. PyTorch **или** TensorFlow. LangChain **или** LlamaIndex. Потом второе.
3. **Воспроизводите статьи руками.** Прочитал статью → реализовал упрощённую версию → понял. Без этого вы её не знаете.
4. **Стройте портфолио с первого месяца.** GitHub, Hugging Face, технический блог. Без портфолио вас не наймут даже джуниором.
5. **Метрика важнее модели.** Сначала придумайте, как измерить успех, потом обучайте. Иначе вы оптимизируете шум.

---

## 🗺️ Структура: 7 треков

| # | Трек | Что освоите | Длительность |
|---|------|-------------|--------------|
| 1 | **Фундамент** | Python, математика, статистика, инструменты | 4–6 нед |
| 2 | **Классический ML** | scikit‑learn, табличные данные, метрики, валидация | 4–6 нед |
| 3 | **Deep Learning** | PyTorch, NN, CV, NLP, тренировочный цикл | 6–8 нед |
| 4 | **LLM и трансформеры** | Внутренности GPT, fine‑tuning, RAG, агенты | 6–10 нед |
| 5 | **Generative AI** | Diffusion, мультимодальность, prompt engineering | 4–6 нед |
| 6 | **MLOps и прод** | Docker, K8s, CI/CD, мониторинг, vLLM, serving | 4–6 нед |
| 7 | **Специализация** | CV / NLP / RecSys / RL / Safety — на выбор | 8+ нед |

---

## 🧱 Трек 1 — Фундамент (Python + математика)

**Цель:** перестать бояться формул и научиться писать код, который читают другие.

**Темы:**

- Python: типы, функции, классы, dataclasses, генераторы, `async/await`, type hints.
- Стек данных: `numpy`, `pandas`, `polars`, `matplotlib`, `seaborn`, `plotly`.
- Математика: линейная алгебра (векторы, матрицы, SVD), производные, градиенты, цепное правило.
- Теорвер и статистика: распределения, ЦПТ, MLE, доверительные интервалы, A/B тесты, p‑value.
- Инструменты: Jupyter, VS Code, Git, GitHub, виртуальные окружения (`uv`, `poetry`, `conda`).

**Артефакт:** ноутбук с EDA на реальном датасете (Kaggle / Open Data) + краткий отчёт «что я увидел в данных».

**Готов к следующему треку, когда:**

- Можете объяснить разницу между корреляцией и причинно‑следственной связью.
- Не путаете `loc` и `iloc` в pandas.
- Знаете, что такое градиент, и как он вычисляется для простой функции вручную.

---

## 📊 Трек 2 — Классический ML

**Цель:** научиться решать задачи без нейросетей. Большинство реальных задач в индустрии — это табличные данные.

**Темы:**

- Подготовка данных: пропуски, выбросы, feature engineering, encoding, скейлинг.
- Алгоритмы: линейная и логистическая регрессии, KNN, деревья решений, Random Forest, **градиентный бустинг (XGBoost, LightGBM, CatBoost)**.
- Кластеризация и снижение размерности: K‑Means, DBSCAN, PCA, t‑SNE, UMAP.
- Метрики: precision/recall/F1, ROC‑AUC, PR‑AUC, MAE/RMSE, MAPE — **и когда какую брать**.
- Валидация: train/val/test, K‑Fold, stratified, time‑series split, data leakage.
- Интерпретируемость: feature importance, SHAP, partial dependence.

**Артефакт:** end‑to‑end Kaggle‑соревнование (любое классическое — Titanic не считается). Финал — отчёт с метриками, ошибками модели и идеями улучшений.

**Готов к следующему треку, когда:**

- Понимаете, почему `ROC‑AUC` может вводить в заблуждение на дисбалансе.
- Знаете, что такое data leakage, и 3 способа его словить.
- Можете объяснить, почему градиентный бустинг бьёт нейросети на табличных данных.

---

## 🧠 Трек 3 — Deep Learning

**Цель:** перестать смотреть на нейросети как на чёрный ящик. Уметь писать тренировочный цикл с нуля.

**Темы:**

- PyTorch: тензоры, autograd, `nn.Module`, `DataLoader`, `optimizer`, `loss`.
- Базовые сети: MLP, CNN, RNN/LSTM/GRU.
- Тренировочный цикл: train/eval, чекпоинты, ранняя остановка, learning rate scheduler.
- Регуляризация: dropout, weight decay, batch norm, layer norm, data augmentation.
- Оптимизаторы: SGD, momentum, Adam, AdamW. Что делает warmup.
- CV: ResNet, EfficientNet, transfer learning, fine‑tuning.
- NLP до трансформеров: word2vec, GloVe, embeddings, seq2seq.

**Артефакт:** **mini‑GPT (~10M параметров) с нуля на PyTorch** + бенчмарк против `torch.nn.MultiheadAttention`. Это закроет понимание attention раз и навсегда.

**Готов к следующему треку, когда:**

- Можете написать тренировочный цикл с нуля без копипасты.
- Знаете, почему vanishing gradients ломают глубокие сети и что с этим делают.
- Понимаете разницу между batch norm и layer norm, и почему трансформеры используют именно layer norm.

---

## 🚀 Трек 4 — LLM и трансформеры

**Цель:** понимать, как работают GPT‑подобные модели **внутри**, и уметь их применять в проде.

**Темы:**

- **Архитектура трансформера:** attention, self‑attention, multi‑head, positional encoding (sinusoidal, RoPE, ALiBi).
- **Внутренности:** KV‑cache, MQA/GQA, FlashAttention, speculative decoding, continuous batching, paged attention (vLLM).
- **Tokenization:** BPE, WordPiece, SentencePiece. Почему `tiktoken` важен.
- **Pre‑training → SFT → RLHF/DPO:** как из «чтения интернета» получается ChatGPT.
- **Prompt engineering:** few‑shot, chain‑of‑thought, ReAct, structured output (JSON mode, function calling).
- **RAG:** chunking, embeddings, vector DB (Qdrant, Weaviate, pgvector), re‑ranking, hybrid search.
- **Fine‑tuning:** LoRA, QLoRA, PEFT, DPO. Когда дообучать, а когда хватит промпта.
- **AI‑агенты:** ReAct, tool use, function calling, MCP (Model Context Protocol), multi‑agent.

**Артефакт:** **свой RAG‑сервис** на корпусе документов (книги / документация / Telegram‑архив) + дообучение open‑source модели (Llama / Qwen / Mistral 7B) через LoRA на собственном датасете.

**Готов к следующему треку, когда:**

- Объясняете KV‑cache на пальцах, и как он влияет на latency и память.
- Понимаете, в чём разница между PPO и DPO.
- Знаете, когда RAG лучше fine‑tuning, и наоборот.

---

## 🎨 Трек 5 — Generative AI и мультимодальность

**Цель:** уметь генерировать изображения, видео, аудио — и понимать, как это работает.

**Темы:**

- **Diffusion models:** DDPM, DDIM, Latent Diffusion (Stable Diffusion), Flow Matching, DiT (Sora‑like).
- **Управление генерацией:** classifier‑free guidance, ControlNet, LoRA для диффузии, IP‑Adapter.
- **VLM (Vision‑Language Models):** CLIP, BLIP, LLaVA, Qwen‑VL, нативные мультимодальные (Gemini, GPT‑4o).
- **Аудио:** Whisper, TTS, voice cloning, audio diffusion.
- **Prompt engineering для генеративки:** стиль, композиция, негативные промпты.

**Артефакт:** свой Stable Diffusion XL, дообученный через LoRA на собственном датасете (стиль / лицо / объект), + Gradio‑демо на Hugging Face Spaces.

---

## ⚙️ Трек 6 — MLOps и production

**Цель:** довести модель до прода и не разбудить дежурного ночью.

**Темы:**

- Docker, docker‑compose, multi‑stage builds.
- Kubernetes: pods, services, deployments, HPA. Helm‑чарты.
- CI/CD: GitHub Actions, тесты моделей, автоматический деплой.
- Serving: **vLLM**, **TGI** (Hugging Face), **TensorRT‑LLM**, **llama.cpp**, **Ollama** (для LLM); BentoML, Ray Serve (для классики).
- Мониторинг: метрики качества, drift detection, latency, токены/сек. Evidently, Grafana + Prometheus.
- Эксперименты: **Weights & Biases**, **MLflow**, версионирование данных (**DVC**).
- LLM‑observability: **LangSmith**, **Langfuse**, **Arize Phoenix**.

**Артефакт:** свой LLM‑сервис в Docker → Kubernetes → с автоскейлингом, мониторингом и health‑checks. Метрики: tokens/sec, p99 latency, % ошибок.

---

## 🎯 Трек 7 — Специализация (на выбор 1–2)

К этому моменту у вас есть фундамент. Дальше — **глубина** в одной из областей:

- **NLP / LLM Engineer** — fine‑tuning, RAG в проде, агенты, LLM‑evaluation.
- **Computer Vision** — детекция, сегментация, диффузия, видео, 3D, медицинский CV.
- **Recommender Systems** — collaborative filtering, two‑tower, ранкеры, RecSys в проде.
- **Reinforcement Learning** — Q‑learning, policy gradients, PPO, RLHF, агенты в средах.
- **AI Safety / Alignment** — red‑teaming, evaluation, interpretability, guardrails.
- **MLOps / Platform** — инфраструктура для ML‑команды, GPU‑оркестрация, feature stores.

---

## 📐 Уровни: junior → middle → senior → guru

| Уровень | Что умеет |
|---------|-----------|
| **Junior ML** | Решает табличные задачи, тренирует CNN на готовых датасетах, понимает метрики, читает чужие ноутбуки. |
| **Middle ML** | Пишет тренировочный цикл с нуля, дообучает LLM, поднимает RAG, понимает evaluation, делает прод. |
| **Senior ML / LLM Engineer** | Архитектура ML‑систем, выбор моделей и инфраструктуры, mentoring, исследовательские развилки. |
| **Guru / Vibe coder с пониманием** | Объясняет, как работает FlashAttention; реализует DPO, speculative decoding, кастомные ядра. Пишет свои статьи / open‑source. |

---

## 💰 Лучшие платные курсы

- **[Stepik — C# с нуля до профи](https://stepik.org/a/282984/pay?promo=4b3c5f3000f16022)** — ООП, SOLID, LINQ, async/await, DI, EF Core, ASP.NET Core, Docker, Kubernetes. Если параллельно с ML вы укрепляете инженерный фундамент — это лучший русскоязычный курс по C#: всё, что казалось магией, становится рабочим инструментом.

> 💡 По ML платные курсы добавим по мере появления. Пока сильнейшая бесплатная база покрывает 90% потребностей — см. ниже.

---

## 🆓 Лучшие бесплатные курсы по ML / DL / LLM

Этого списка хватит, чтобы стать ML‑инженером без единой копейки. Главное — **доходить до конца** и делать домашки.

### 🟢 Старт: математика и Python

- **[Khan Academy — Linear Algebra / Calculus / Probability](https://www.khanacademy.org/math)** — бесплатно, на пальцах, идеально для входа.
- **[3Blue1Brown — Essence of Linear Algebra / Neural Networks](https://www.3blue1brown.com/)** — визуальные интуитивные ролики. Обязательно.
- **[CS50P — Introduction to Python (Harvard)](https://cs50.harvard.edu/python/)** — лучший вводный курс по Python.

### 🟡 Classical ML

- **[Andrew Ng — Machine Learning Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction)** — классика, аудит бесплатно.
- **[StatQuest with Josh Starmer (YouTube)](https://www.youtube.com/@statquest)** — ML и статистика на пальцах с песнями. Серьёзно — лучший канал для интуиции.
- **[Open Machine Learning Course (ODS.ai / mlcourse.ai)](https://mlcourse.ai/)** — главный русскоязычный открытый курс по ML.
- **[ШАД — Школа анализа данных Яндекса (открытые материалы)](https://academy.yandex.ru/handbook/ml)** — учебник по ML от Яндекса, бесплатно.

### 🔴 Deep Learning

- **[fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/)** — top‑down подход: сначала работает, потом разбираемся. Лучший практический курс.
- **[Andrew Ng — Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)** — фундамент.
- **[Andrej Karpathy — Neural Networks: Zero to Hero (YouTube)](https://karpathy.ai/zero-to-hero.html)** — **обязательно к просмотру**. От `micrograd` до `nanoGPT` своими руками.
- **[CS231n — Stanford CV](http://cs231n.stanford.edu/)** — классический курс по computer vision.
- **[Dive into Deep Learning (d2l.ai)](https://d2l.ai/)** — бесплатный интерактивный учебник с кодом.
- **[Deep Learning School (МФТИ)](https://www.dlschool.org/)** — лучший русскоязычный курс по DL.

### 🟣 LLM, трансформеры и Generative AI

- **[Hugging Face — NLP Course](https://huggingface.co/learn/nlp-course)** — официальный курс по работе с трансформерами через библиотеку Transformers. Полное прохождение pipeline от токенизации до деплоя.
- **[Hugging Face — Deep RL Course](https://huggingface.co/learn/deep-rl-course)** — RL с практикой.
- **[Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course)** — как работают Stable Diffusion и Flux.
- **[Hugging Face — Audio Course](https://huggingface.co/learn/audio-course)** — Whisper, TTS, обработка звука.
- **[Hugging Face — Agents Course](https://huggingface.co/learn/agents-course)** — официальный курс по AI‑агентам, smolagents, LangGraph. Самый актуальный материал 2025.
- **[DeepLearning.AI — Short Courses](https://www.deeplearning.ai/short-courses/)** — десятки бесплатных коротких курсов от Andrew Ng в партнёрстве с OpenAI, Anthropic, LangChain, LlamaIndex.
- **[Full Stack Deep Learning — LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)** — двухдневный буткамп по построению LLM‑приложений. Полностью бесплатно на YouTube.
- **[Stanford CS25 — Transformers United](https://web.stanford.edu/class/cs25/)** — гостевые лекции от авторов главных работ по трансформерам (включая авторов «Attention Is All You Need»).
- **[Maxime Labonne — LLM Course (GitHub)](https://github.com/mlabonne/llm-course)** — структурированный roadmap по LLM с тетрадями для fine‑tuning, quantization, evaluation. Один из самых популярных open‑source курсов 2024–2025.

### 🟠 Prompt engineering, RAG, AI‑агенты

- **[Anthropic — Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)** — официальный туториал от Anthropic по работе с Claude. Лучший источник по prompt engineering.
- **[Anthropic — Courses (GitHub)](https://github.com/anthropics/courses)** — полный набор бесплатных курсов: API Fundamentals, Prompt Engineering, Real World Prompting, Tool Use, Model Context Protocol.
- **[OpenAI Cookbook](https://cookbook.openai.com/)** — сотни рабочих примеров от OpenAI: prompt engineering, function calling, embeddings, RAG.
- **[Microsoft — Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners)** — 21 урок с кодом по построению GenAI‑приложений. Полностью бесплатно.
- **[Microsoft — AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)** — официальный курс Microsoft по AI‑агентам, 10+ уроков с примерами.
- **[LangChain Academy](https://academy.langchain.com/)** — бесплатные курсы по LangChain и LangGraph: Introduction to LangGraph, Intro to LangSmith.
- **[Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/)** — сводный каталог техник промптинга, бесплатно онлайн.

### 🟤 MLOps, production и инфраструктура

- **[Made With ML — MLOps Course](https://madewithml.com/courses/mlops/)** — полный курс по доведению ML до прода: design, develop, deploy, iterate. Бесплатно.
- **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)** — производственный ML от Berkeley. Все лекции на YouTube бесплатно.
- **[MLOps Zoomcamp (DataTalks.Club)](https://github.com/DataTalksClub/mlops-zoomcamp)** — практический буткамп по MLOps. Бесплатно, с домашками и сертификатом.
- **[Machine Learning Engineering for Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)** — Andrew Ng + Robert Crowe, бесплатный аудит.
- **[Designing Machine Learning Systems — Chip Huyen (заметки и материалы)](https://huyenchip.com/ml-interviews-book/)** — бесплатные материалы автора главной книги по ML‑системам.

### ⚫ Reinforcement Learning и продвинутые темы

- **[David Silver — Reinforcement Learning Course (DeepMind, UCL)](https://www.davidsilver.uk/teaching/)** — классический курс по RL от автора AlphaGo. Все лекции на YouTube.
- **[Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)** — официальное введение в Deep RL от OpenAI. Теория + рабочий код.

> 🎯 **Как использовать список:** не пытайтесь пройти всё. Возьмите по 1 курсу из каждого блока, который актуален вашему текущему треку. Пройдите до конца — с домашками и проектом. Потом возвращайтесь за следующим.

---

## ❓ FAQ

**Сколько нужно математики, чтобы войти в ML?**
Линейная алгебра на уровне умножения матриц, производные и градиенты, базовый теорвер и статистика. Дифференциальную геометрию учить **не нужно**. Если плаваете — Khan Academy + 3Blue1Brown за месяц закроют.

**PyTorch или TensorFlow?**
PyTorch — де‑факто стандарт в 2025. Начинайте с него. JAX — для исследований, если пойдёте глубоко в research.

**Брать ли платный курс или хватит бесплатных?**
Бесплатных курсов выше хватит на путь от нуля до middle. Платные курсы оправданы, если вам нужна структура, дедлайны и ментор. Сертификат сам по себе не нанимает — нанимает портфолио.

**Сколько времени до первого оффера?**
6–12 месяцев интенсивной работы (15+ часов в неделю) с фокусом на портфолио. Меньше — нереалистично. Больше — нормально, если параллельно работаете.

**Что делать, если статьи на arXiv пока непонятны?**
Это нормально. Читайте сначала разборы (Lilian Weng, Jay Alammar, paperswithcode.com), потом саму статью. Через 30 прочитанных статей пойдёт легче.

**LLM — это пузырь, или будущее?**
LLM — это инструмент, который останется. Конкретные продукты вокруг них могут меняться, но навык работы с трансформерами, RAG, агентами и fine‑tuning будет востребован минимум 5–10 лет.

---

## 🤝 Contributing

PR с уточнениями, обновлениями ссылок, новыми ресурсами и опытом приветствуются. Перед отправкой:

- Один PR — одна логическая правка.
- Сохраняйте тон: трезвый, прикладной, без маркетинга.
- Ресурсы добавляем только те, которые сами проверяли.

---

## 📄 License

MIT. Используйте, форкайте, адаптируйте под свои команды и студии.

---

> Этот roadmap по машинному обучению — карта местности, а не маршрут. Маршрут вы прокладываете сами, исходя из задач, рынка и того, что вас зажигает. Удачи на пути от первого `import numpy` до собственной обученной LLM.
