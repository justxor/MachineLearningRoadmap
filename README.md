# 🤖 Machine Learning Roadmap: от базы до гуру вайбкодинга

> **Карта обучения машинному обучению (Machine Learning, Deep Learning, LLM, Generative AI, MLOps)** — от первого `import numpy` до уровня инженера, который понимает, **как ИИ работает внутри**, и может писать прод‑системы, а не только дёргать API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Roadmap](https://img.shields.io/badge/Roadmap-2025--2026-blue.svg)](#)
[![Made for](https://img.shields.io/badge/Made%20for-RU%20ML%20community-red.svg)](#)

**Ключевые слова:** машинное обучение, глубокое обучение, ML roadmap, LLM, нейросети, PyTorch, transformers, RAG, AI агенты, MLOps, fine‑tuning, prompt engineering, vibe coding, generative AI, Hugging Face, дата‑сайентист, ML инженер, как стать ML инженером, обучение машинному обучению с нуля.

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

## 📺 Полезные Telegram‑каналы (читать каждый день)

Подборка каналов, которые реально помогают держать руку на пульсе индустрии: свежие статьи, релизы моделей, разборы архитектур, вакансии и собеседования. Подписывайтесь точечно и читайте регулярно — это даёт больше, чем разовый «забег» по курсам.

### 🤖 Машинное обучение, нейросети и LLM

- **[@ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data)** — **главный русскоязычный канал по ML/AI/Big Data**. Свежие статьи, релизы моделей, разборы.
- **[Data Analysis / ML](https://t.me/data_analysis_ml)** — дата‑аналитика и ML без воды: туториалы, библиотеки, кейсы.
- **[Вистехно](https://t.me/vistehno)** — про технологии, AI и инженерную культуру.
- **[Machine Learning Interview](https://t.me/machinelearning_interview)** — задачи, разборы собесов и теоретические вопросы по ML.
- **[Data Science / IoT](https://t.me/datascienceiot)** — Data Science, индустриальные применения и IoT.
- **[Artificial Intelligence / DL](https://t.me/ArtificialIntelligencedl)** — обзоры статей по deep learning и AI.
- **[Machine Learning Test](https://t.me/Machinelearningtest)** — тесты, мини‑задачи и проверка знаний по ML.
- **[Machine Learning](https://t.me/machinee_learning)** — англоязычные новости и материалы по ML.
- **[Machine Learning RU](https://t.me/machinelearning_ru)** — русскоязычный канал по ML, статьи и инструменты.
- **[Neural Networks](https://t.me/neural)** — про нейронные сети, архитектуры и применения.
- **[Machine Learning Rus](https://t.me/machinelearning_rus)** — материалы по ML на русском, разборы и подборки.
- **[Big Data AI](https://t.me/bigdatai)** — Big Data, аналитика и AI‑инструменты для работы с данными.
- **[@ai_generative](https://t.me/ai_generative)** — генеративный AI: LLM, диффузионные модели, image/video/audio generation.

### 📚 Книги, базы данных и SQL

- **[Machine Learning Books](https://t.me/machinelearning_books)** — книги, гайды и учебные материалы по ML/AI.
- **[SQL Hub](https://t.me/sqlhub)** — SQL, оптимизация запросов и работа с реляционными БД.
- **[Databases](https://t.me/databases_tg)** — про базы данных: реляционные, NoSQL, аналитические.

### 💼 Вакансии и карьера

- **[Data Science / ML Jobs](https://t.me/datascienceml_jobs)** — вакансии в Data Science и ML, удалёнка и офис.
- **[Machine Learning Jobs](https://t.me/Machinelearning_Jobs)** — отдельная лента ML‑вакансий: junior, middle, senior, research.

### 📁 Папки и оптовая подписка

- **[📁 Большая папка ML/AI каналов](https://t.me/addlist/u15AMycxRMowZmRi)** — кураторская подборка лучших каналов по машинному обучению, нейросетям, LLM и MLOps. Подписаться оптом.

> 💡 Совет: не подписывайтесь на 200 каналов. Возьмите 5–7 ключевых, читайте каждый день 15 минут — этого хватит, чтобы быть в курсе индустрии. Остальные держите в отдельной папке и заглядывайте раз в неделю.

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

## 🧠 Продвинутые темы (deep dive)

Этот блок — не «обязательная программа», а карта **специализаций**. Выбирайте 1–2 направления после трека 5–6, погружайтесь до уровня, на котором можете писать свои реализации, а не только дёргать API.

### 🔴 Внутренности трансформера: от математики до CUDA

- **Что должен уметь объяснить «гуру вайбкодинга»:**
  - Почему `attention` — это soft k-NN, а не «магия».
  - Что такое KV-cache, как он экономит compute и почему ломается на длинных контекстах.
  - Чем отличаются **MHA / MQA / GQA**, зачем придумали **RoPE**, **ALiBi**, **YaRN**.
  - Что такое **FlashAttention** (1/2/3), почему быстрее наивного `softmax(QK^T)V`.
  - Как работает **speculative decoding**, **continuous batching**, **paged attention** (vLLM).
- **Артефакт:** свой mini-GPT (~10M параметров) с нуля на PyTorch + бенчмарк против `torch.nn.MultiheadAttention`.
- **Ресурсы:** Karpathy `nanoGPT` и `build-nanogpt`, статьи **Attention Is All You Need**, **GPT-2/3/4 tech reports**, блог Lilian Weng.

### 🔴 Alignment, RLHF, DPO и пост-тренинг LLM

- **Темы:** SFT → Reward Modeling → PPO/RLHF → DPO/IPO/KTO → Constitutional AI → RLAIF.
- **Понимать на пальцах:**
  - Почему «просто SFT» недостаточно для chat-моделей.
  - В чём математическая разница между PPO и DPO (и почему DPO выиграл по простоте).
  - Что такое **reward hacking**, **sycophancy**, **mode collapse** после RLHF.
- **Артефакт:** дообучение open-source модели (Llama / Qwen / Mistral 7B) через **LoRA + DPO** на своём датасете, сравнение метрик с базовой моделью.
- **Ресурсы:** **InstructGPT paper**, **DPO paper (Rafailov et al.)**, **Anthropic Constitutional AI**, библиотеки `trl`, `axolotl`, `unsloth`.

### 🔴 Diffusion models и генеративка изображений/видео

- **Темы:** DDPM → DDIM → Score-based models → Latent Diffusion (Stable Diffusion) → Flow Matching → Rectified Flow → DiT (Sora-like).
- **Понимать:** прямой/обратный процесс, **classifier-free guidance**, **ControlNet**, **LoRA для диффузии**, **IP-Adapter**.
- **Артефакт:** свой DDPM на MNIST/CIFAR с нуля + fine-tune Stable Diffusion XL через LoRA на собственном датасете (стиль, лицо, объект).
- **Ресурсы:** курс **fast.ai Part 2 (Diffusion from scratch)**, статьи **DDPM**, **LDM**, **DiT**, блог Sander Dieleman.

### 🔴 Мультимодальность и VLM

- **Темы:** CLIP → BLIP/BLIP-2 → LLaVA → Qwen-VL → GPT-4V → нативные мультимодальные модели (Gemini, GPT-4o).
- **Понимать:** как картинка превращается в токены, что такое **vision encoder + projector + LLM**, почему OCR-задачи всё ещё ломаются.
- **Артефакт:** свой VLM-стек: CLIP-эмбеддинги → projection layer → LLM, дообученный на узком домене (медснимки, мемы, схемы).

### 🔴 AI-агенты и tool use на проде

- **Темы:** ReAct → Toolformer → function calling → **MCP (Model Context Protocol)** → multi-agent (CrewAI, AutoGen, LangGraph) → computer use агенты (Claude Computer Use, OpenAI Operator).
- **Понимать:** почему **«один большой промпт» не масштабируется**, что такое **state machine для агента**, как ловить и чинить **infinite loops** и **hallucinated tools**.
- **Артефакт:** агент-аналитик, который сам ходит в БД, пишет SQL, строит графики и присылает отчёт в Telegram. С полноценным трейсингом через **LangSmith / Langfuse / Phoenix**.

### 🔴 LLM evaluation — самое недооценённое

- **Темы:** academic benchmarks (MMLU, HellaSwag, GSM8K, HumanEval, BBH, MT-Bench, Arena-Hard) → **task-specific eval** → **LLM-as-a-Judge** → **golden datasets** → **A/B на проде**.
- **Понимать:** почему **«вайб-чек»** — это не evaluation, что такое **contamination**, как считать **pass@k**, **win rate**, **factuality**.
- **Артефакт:** свой eval-харнесс для конкретной задачи (RAG / classification / agent) с автоматическим прогоном на каждом коммите.
- **Инструменты:** `lm-evaluation-harness`, `OpenAI evals`, `promptfoo`, `DeepEval`, `Ragas`, `TruLens`.

### 🔴 Безопасность, jailbreaks, red-teaming

- **Темы:** prompt injection, indirect prompt injection, data exfiltration, jailbreaks (DAN, GCG, many-shot), **PII leakage**, **model stealing**, **membership inference**.
- **Понимать:** **OWASP Top-10 for LLM Applications**, разницу между **alignment** и **safety**, что такое **defence in depth** для LLM-приложений.
- **Артефакт:** red-team отчёт по своему RAG-сервису + набор guardrails (input/output фильтры, rate limits, PII-маски).
- **Ресурсы:** **OWASP LLM Top-10**, **Anthropic responsible scaling policy**, гайды **NIST AI RMF**.

### 🔴 Эффективность: квантование, distillation, edge

- **Темы:** PTQ vs QAT, GPTQ, AWQ, GGUF, bitsandbytes, **knowledge distillation**, **pruning**, **MoE**.
- **Понимать:** где теряется качество при int4/int8, когда выгоднее меньшая модель + RAG, чем большая «в лоб».
- **Артефакт:** свой LLM, запущенный на ноутбуке/телефоне через **llama.cpp** / **MLX** / **ONNX Runtime**, бенчмарк tokens/sec и качества.

---

## 📚 Must-read papers (минимальный канон)

Если вы можете рассказать **своими словами**, что в этих статьях и зачем — вы понимаете, как работает современный AI изнутри.

**Фундамент трансформеров и LLM:**

- **Attention Is All You Need** (Vaswani et al., 2017) — оригинал трансформера.
- **BERT** (Devlin et al., 2018) — masked LM, эпоха encoder-only.
- **GPT-2 / GPT-3 / GPT-4 technical reports** — scaling laws на практике.
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) и **Chinchilla** (Hoffmann et al., 2022) — сколько данных vs параметров.
- **LoRA** (Hu et al., 2021) — почему дообучение стало дешёвым.
- **FlashAttention 1/2** (Dao et al.) — IO-aware attention.

**Alignment и пост-тренинг:**

- **InstructGPT** (Ouyang et al., 2022) — RLHF в проде.
- **Constitutional AI** (Bai et al., 2022) — Anthropic, RLAIF.
- **Direct Preference Optimization** (Rafailov et al., 2023) — DPO без reward model.
- **Self-Instruct** / **Alpaca** — синтетические данные для SFT.

**RAG, агенты, tool use:**

- **Retrieval-Augmented Generation** (Lewis et al., 2020).
- **ReAct** (Yao et al., 2022) — reasoning + acting.
- **Toolformer** (Schick et al., 2023).
- **Chain-of-Thought Prompting** (Wei et al., 2022) и **Tree of Thoughts**.

**Генеративка и мультимодальность:**

- **DDPM** (Ho et al., 2020) — denoising diffusion.
- **Latent Diffusion** (Rombach et al., 2022) — Stable Diffusion.
- **CLIP** (Radford et al., 2021).
- **DiT** (Peebles & Xie, 2023) — diffusion transformer (Sora-like).

**Состояние индустрии (читать обзоры раз в полгода):**

- **State of AI Report** (Nathan Benaich) — ежегодно.
- **Stanford AI Index Report** — ежегодно.
- **A Survey of Large Language Models** (Zhao et al.) — обновляется.

> 💡 Совет: читайте статьи **с кодом рядом**. Если статья без репозитория и реализации — её влияние обычно переоценено.

---

## 📖 Книги, которые реально меняют уровень

**Базовый ML/DL:**

- **«Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow»** — Aurélien Géron. Лучшая практическая книга для входа.
- **«Deep Learning»** — Goodfellow, Bengio, Courville. Теоретический фундамент. Бесплатно онлайн.
- **«Pattern Recognition and Machine Learning»** — Christopher Bishop. Для тех, кто хочет математику глубоко.
- **«The Elements of Statistical Learning»** — Hastie, Tibshirani, Friedman. Бесплатно онлайн, классика статистики.
- **«Probabilistic Machine Learning»** — Kevin Murphy (2 тома). Современная альтернатива Бишопу. Бесплатно онлайн.

**Продакшен и инженерия:**

- **«Designing Machine Learning Systems»** — Chip Huyen. Главная книга по ML-системам на проде.
- **«Machine Learning Engineering»** — Andriy Burkov. Прикладная, без воды.
- **«Building Machine Learning Powered Applications»** — Emmanuel Ameisen.
- **«Reliable Machine Learning»** — O'Reilly, SRE-подход к ML.

**LLM и Generative AI:**

- **«Build a Large Language Model (From Scratch)»** — Sebastian Raschka. Свой GPT с нуля построчно.
- **«Hands-On Large Language Models»** — Jay Alammar, Maarten Grootendorst.
- **«AI Engineering»** — Chip Huyen (2025) — про построение LLM-приложений.
- **«Generative Deep Learning»** — David Foster.

**Софт-скиллы и мышление:**

- **«The Hundred-Page Machine Learning Book»** — Andriy Burkov. Идеально для быстрого повторения.
- **«Storytelling with Data»** — Cole Nussbaumer Knaflic. Как доносить результаты до бизнеса.

---

## 🛠️ Боевой стек инструментов (cheat sheet)

Каждый инструмент — выучить до уровня «знаю команды наизусть и могу объяснить trade-offs».

**Данные и эксперименты:**

- **Обработка:** `pandas`, `polars`, `duckdb`, `pyarrow`, `dask`.
- **Визуализация:** `matplotlib`, `seaborn`, `plotly`, `altair`.
- **Эксперименты:** **Weights & Biases**, **MLflow**, **Neptune**, **ClearML**.
- **Версионирование данных:** **DVC**, **lakeFS**, **Delta Lake**.

**Модели и тренировка:**

- **Фреймворки:** `PyTorch` (де-факто стандарт), `JAX` (для исследований), `scikit-learn` (classic ML).
- **Высокий уровень:** `PyTorch Lightning`, `Hugging Face Transformers`, `Accelerate`.
- **Дообучение LLM:** `trl`, `peft`, `unsloth`, `axolotl`, `LLaMA-Factory`.
- **Распределённое:** `DeepSpeed`, `FSDP`, `Megatron-LM`.

**Инференс и серинг:**

- **LLM серинг:** **vLLM**, **TGI** (Hugging Face), **SGLang**, **TensorRT-LLM**, **llama.cpp**, **Ollama**, **LM Studio**.
- **Классические модели:** `BentoML`, `Ray Serve`, `Triton Inference Server`, `TorchServe`.
- **Edge / on-device:** **ONNX Runtime**, **CoreML**, **MLX** (Apple Silicon), **TensorFlow Lite**.

**LLM-приложения:**

- **Оркестрация:** **LangChain**, **LlamaIndex**, **LangGraph**, **Haystack**, **DSPy**.
- **Векторные БД:** **Qdrant**, **Weaviate**, **Milvus**, **pgvector**, **Chroma**, **FAISS**.
- **Observability:** **LangSmith**, **Langfuse**, **Arize Phoenix**, **Helicone**.
- **Evaluation:** **Ragas**, **DeepEval**, **promptfoo**, **TruLens**.
- **Guardrails:** **NeMo Guardrails**, **Guardrails AI**, **Llama Guard**.

**MLOps и инфраструктура:**

- **Оркестрация:** **Airflow**, **Prefect**, **Dagster**, **Kubeflow**.
- **Контейнеры/облако:** **Docker**, **Kubernetes**, **Terraform**, **AWS/GCP/Azure**.
- **Мониторинг:** **Evidently**, **WhyLabs**, **Grafana + Prometheus**.
- **Feature store:** **Feast**, **Tecton**.

> ⚠️ **Не учите всё сразу.** Берите 1 инструмент из каждой категории под текущий проект. Остальное — на bookmark.

---

## 🌐 Сообщества и где быть «в курсе»

**Англоязычные:**

- **Hugging Face** ([huggingface.co](https://huggingface.co)) — модели, датасеты, Spaces, форум.
- **Papers with Code** ([paperswithcode.com](https://paperswithcode.com)) — статьи + бенчмарки + код.
- **arXiv** ([arxiv.org](https://arxiv.org)) — секции `cs.LG`, `cs.CL`, `cs.CV`, `stat.ML`.
- **r/MachineLearning**, **r/LocalLLaMA** — Reddit, лучшие практические треды по open-source LLM.
- **EleutherAI Discord**, **Hugging Face Discord** — где сидят авторы статей.
- **AlphaSignal**, **The Batch (DeepLearning.AI)**, **Import AI** (Jack Clark), **Interconnects** (Nathan Lambert) — рассылки.

**Русскоязычные:**

- **ODS.ai** ([ods.ai](https://ods.ai)) — главное русскоязычное ML-сообщество, Slack + митапы.
- **Data Fest** — ежегодная конференция ODS.
- **Kaggle ru community** — Telegram-чаты по соревнованиям.
- **Telegram-каналы** — см. блок «Полезные Telegram-каналы» в начале README.

**Конференции (что смотреть на YouTube, если не попасть очно):**

- **NeurIPS**, **ICML**, **ICLR** — топ-3 академические.
- **ACL**, **EMNLP**, **NAACL** — NLP.
- **CVPR**, **ICCV**, **ECCV** — computer vision.
- **MLSys** — ML-системы и инфраструктура.
- **KDD** — applied data mining.
- **Data Council**, **MLOps World** — индустриальные.

---

## 🏆 Соревнования и портфолио

**Где набивать руку:**

- **Kaggle** — классика. Цель не «золото», а **публичные ноутбуки** с разбором решений топов.
- **Hugging Face competitions** — фокус на LLM/мультимодальность.
- **AIcrowd**, **DrivenData**, **Zindi** — задачи с социальным импактом.
- **Numerai**, **QuantConnect** — если интересен финтех.
- **LMSYS Chatbot Arena** — посылать свои fine-tuned модели на public eval.

**Что должно быть в портфолио к моменту найма на ML/LLM позицию:**

1. **3–5 end-to-end проектов** на GitHub: README с метриками, демо (HF Space / Streamlit / Gradio), воспроизводимый код.
2. **1 хардовый проект:** своя реализация чего-то нетривиального (mini-GPT, DDPM, RAG-сервис, агент) — **не туториал**.
3. **Технический блог** — 5–10 постов про свои эксперименты. Habr / Medium / личный сайт.
4. **Вклад в open-source** — хотя бы 1–2 merged PR в популярные репозитории (transformers, langchain, vllm и т.п.).
5. **Профиль на Hugging Face** — выложенные модели/датасеты/Spaces.

---

## 📊 Как измерять свой прогресс (без самообмана)

Простые контрольные вопросы для каждого уровня. Если не можете ответить **без гугла за 60 секунд** — уровень не пройден.

**Junior ML:**

- В чём разница между bias и variance? Как их балансировать?
- Когда L1 регуляризация лучше L2?
- Почему ROC-AUC может вводить в заблуждение на дисбалансе?
- Что такое data leakage и 3 способа его словить?

**Middle ML / DL:**

- Почему vanishing gradients ломают глубокие сети и что с этим делают?
- В чём идея batch norm и почему он не всегда работает в трансформерах?
- Что происходит при `model.eval()` в PyTorch?
- Чем отличается attention от self-attention? Что такое причинная маска?

**Senior / LLM Engineer:**

- Объясните KV-cache на пальцах. Как он влияет на latency и память?
- В чём математическая разница между PPO и DPO?
- Когда RAG лучше fine-tuning, и наоборот?
- Как бы вы построили eval для чат-бота поддержки **без** human annotators?
- Что вы сделаете, если у RAG-сервиса вдруг упало качество в проде?

**Guru (вайбкодер с пониманием):**

- Напишите псевдокод FlashAttention и объясните, где экономия.
- Почему DPO теоретически эквивалентен PPO при определённых условиях?
- Как реализовать speculative decoding с нуля?
- Спроектируйте архитектуру multi-tenant LLM-сервиса на 10k RPS с SLA 99.9%.

---

## 🗺️ Карьерные треки внутри ML

ML — это не одна профессия. Понимайте, **куда именно** вы целитесь.

- **Data Scientist** — гипотезы, A/B, статистика, бизнес-метрики. Меньше кода, больше коммуникации.
- **ML Engineer** — пайплайны, инференс, latency, надёжность. Ближе к backend.
- **MLOps / Platform Engineer** — инфраструктура для ML-команды. Kubernetes, observability, CI/CD моделей.
- **Research Engineer** — реализация статей, эксперименты с архитектурами. Мост между research и prod.
- **Research Scientist** — свои статьи, PhD-уровень. Топовые лабы: Anthropic, OpenAI, DeepMind, Meta FAIR.
- **LLM / GenAI Engineer** — новая роль. Промпты, RAG, агенты, fine-tuning. Самая горячая в 2024–2026.
- **Applied AI Engineer** — встраивание AI-фич в продукт. Гибрид product + ML + frontend/backend.
- **AI Safety / Alignment Researcher** — red-teaming, evaluation, interpretability. Anthropic, Apollo, METR.

> 🎯 Совет: на джуне нормально быть «универсалом». К мидлу выберите 1–2 трека и копайте вглубь. На сениоре T-shape: один трек глубоко + смежные на уровне «могу собеседовать».


---

> Этот roadmap по машинному обучению — карта местности, а не маршрут. Маршрут вы прокладываете сами, исходя из задач, рынка и того, что вас зажигает. Удачи на пути от первого `import numpy` до собственной обученной LLM.
