# 🤖 Machine Learning Roadmap: от базы до гуру вайбкодинга

> **Карта обучения машинному обучению (Machine Learning, Deep Learning, LLM, Generative AI, MLOps)** — от первого `import numpy` до уровня инженера, который понимает, **как ИИ работает внутри**, и может писать прод‑системы, а не только дёргать API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Roadmap](https://img.shields.io/badge/Roadmap-2025--2026-blue.svg)](#)
[![Made for](https://img.shields.io/badge/Made%20for-RU%20ML%20community-red.svg)](#)

**Ключевые слова:** машинное обучение, глубокое обучение, ML roadmap, LLM, нейросети, PyTorch, transformers, RAG, AI агенты, MLOps, fine‑tuning, prompt engineering, vibe coding, generative AI, Hugging Face, дата‑сайентист, ML инженер, как стать ML инженером, обучение машинному обучению с нуля.

---

## ⚡ Быстрый старт: что сделать на этой неделе

Если вы только зашли и не знаете, с чего начать — вот ровно 7 шагов на ближайшие 7 дней. Без них всё остальное в roadmap не сработает.

1. **Поставьте Python 3.12+, VS Code (или Cursor), Git.** 30 минут.
2. **Заведите GitHub-аккаунт и создайте репозиторий `ml-journey`.** В нём будут все ваши учебные проекты. 15 минут.
3. **Зарегистрируйтесь на [Kaggle](https://www.kaggle.com)** и пройдите [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (4 часа, бесплатно).
4. **Подпишитесь на 3 канала из подборки ниже** — `@ai_machinelearning_big_data`, плюс 2 на выбор.
5. **Откройте ноутбук Colab** и запустите свой первый `import torch; torch.tensor([1,2,3])`. Это снимает страх.
6. **Запланируйте 10 часов в неделю в календаре** — конкретные слоты, не «как получится».
7. **Расскажите кому-нибудь, что учитесь ML.** Соцсеть, друг, чат. Публичное обязательство работает.

> 💡 Если вы сделали эти 7 шагов — поздравляю, вы уже впереди 80% тех, кто «собирается учить ML».

---

## 👥 Если вы… (3 типовых старта)

**🧑‍💻 Если вы разработчик (2+ лет опыта):**
Пропускайте Python, идите сразу в математику (если её не было) и классический ML. Ваше преимущество — вы умеете писать код в проде. Сильная сторона на собеседовании — MLOps и интеграция моделей. Целевая позиция через 6 месяцев: **ML Engineer**, не Data Scientist.

**🧑‍🎓 Если вы студент или меняете профессию:**
Идите по roadmap последовательно. Не торопитесь, дайте 12 месяцев. Главное — портфолио и Kaggle-сабмиты. Целевая позиция: **Junior Data Scientist / ML Engineer**. Стажировка в течение обучения почти обязательна.

**🔬 Если вы из науки / аналитики (физика, биология, экономика):**
У вас уже есть математика и работа с данными — это огромный плюс. Учите Python и инженерную часть (Git, Docker, FastAPI). Ваша ниша — **Research Engineer / Applied Scientist**: позиции, где платят за глубокое понимание моделей, а не за «навайбкодить эндпоинт».

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

## 🤖 Vibe coding: как кодить с Claude, ChatGPT и Copilot на уровне сениора

**Vibe coding** (термин Andrej Karpathy, 2025) — стиль разработки, где вы не пишете каждую строчку руками, а ведёте диалог с LLM: формулируете намерение, итеративно правите, читаете diff, гоняете тесты. Код пишет модель, инженер — главный архитектор, ревьюер и носитель контекста. На рынке 2026 это уже не «читерство», а базовый навык: компании оценивают, **насколько эффективно вы умеете работать в паре с AI**, а не просто «знаете Python».

Этот раздел — выжимка того, что реально работает в проде: какие модели и инструменты брать, как ставить задачи, где не наступить на грабли, и куда расти от «генерю функции» до «веду фичу с нуля с агентом».

### 🧠 Большие LLM для кода: что брать в 2026

| Модель | Сильные стороны | Где использовать | Контекст |
|---|---|---|---|
| **Claude Opus 4.x / Sonnet 4.x** (Anthropic) | Лучший «инженерный» рассуждатель, аккуратный с большими кодбазами, сильный tool use, низкая «галлюцинация» API | Рефакторинг, агенты, code review, долгие сессии в Claude Code | 200K+ токенов |
| **GPT-5 / o-series** (OpenAI) | Сильный reasoning, мультимодальность, отличные «one-shot» решения сложных алгоритмических задач | ChatGPT для архитектурных решений, Codex CLI, GitHub Copilot Chat | 256K+ |
| **Gemini 2.x Pro** (Google) | Гигантский контекст (1M+), хорош для анализа целых репозиториев | Чтение больших кодбаз, миграции, поиск по монорепе | 1M+ |
| **DeepSeek-V3 / R1**, **Qwen3-Coder** | Open-weight, можно крутить локально/в своём VPC, сильный код | Self-hosted, sensitive code, бюджетные сценарии | 128K+ |
| **Llama 3.x / 4.x** (Meta) | Open-weight база для fine-tuning, экосистема | On-prem, кастомные кодинг-модели | 128K |

> 💡 Правило большого пальца: **Claude — для долгой инженерной работы и агентов, GPT — для рассуждений и быстрых ответов, Gemini — когда нужно скормить весь репозиторий, open-weight — когда код нельзя отдавать наружу.**

### 🛠️ Инструменты vibe coding (рабочий стек)

**IDE-агенты и интегрированные среды:**
- **[Claude Code](https://www.anthropic.com/claude-code)** — CLI-агент от Anthropic, живёт в терминале, читает/пишет файлы, гоняет команды, держит контекст всего проекта. Топ для серьёзной работы.
- **[Cursor](https://cursor.com)** — IDE на форке VS Code, лучший автокомплит и agent-mode на рынке. Cmd+K для inline-правок, Cmd+L для чата с проектом, Composer для мультифайловых изменений.
- **[Windsurf](https://windsurf.com)** (бывший Codeium) — конкурент Cursor с Cascade-агентом, хорош для крупных рефакторингов.
- **[GitHub Copilot](https://github.com/features/copilot)** + **Copilot Chat / Workspace** — стандарт индустрии, теперь с агентным режимом и многомодельной поддержкой (Claude, GPT, Gemini).
- **[Aider](https://aider.chat)** — open-source CLI-агент с git-интеграцией, каждое изменение = коммит. Любимец инди-разработчиков.
- **[Cline](https://cline.bot)** / **[Roo Code](https://roocode.com)** — open-source агенты для VS Code, работают с любой моделью через API.

**Чат-интерфейсы для архитектуры и обсуждений:**
- **[Claude.ai](https://claude.ai)** — Projects (загрузка контекста), Artifacts (живые превью), Computer Use (агент управляет браузером).
- **[ChatGPT](https://chatgpt.com)** — Canvas для совместного редактирования, Code Interpreter, GPTs под задачу.
- **[T3 Chat](https://t3.chat)**, **[OpenRouter](https://openrouter.ai)** — мультимодельные интерфейсы, дешевле подписок.

**Для агентов и автоматизации:**
- **[LangChain](https://langchain.com)** / **[LlamaIndex](https://llamaindex.ai)** — фреймворки для RAG и агентов.
- **[CrewAI](https://crewai.com)**, **[AutoGen](https://microsoft.github.io/autogen/)** — мульти-агентные системы.
- **[MCP (Model Context Protocol)](https://modelcontextprotocol.io)** — стандарт от Anthropic для подключения инструментов к LLM. Уже поддержан Claude, OpenAI, Cursor.

### 🎯 Как ставить задачи LLM: prompting для кода

**Базовая структура промпта для серьёзной задачи:**

```
КОНТЕКСТ: Что за проект, стек, ограничения.
ЦЕЛЬ: Что нужно получить на выходе (функция/PR/архитектура).
ВВОД/ВЫВОД: Сигнатуры, типы, примеры.
ОГРАНИЧЕНИЯ: Производительность, зависимости, стиль.
ПРИМЕРЫ: 1–2 похожих куска из текущего кода.
КРИТЕРИЙ ГОТОВНОСТИ: Тесты проходят / соответствует ТЗ / ревью OK.
```

**Техники, которые реально работают:**

1. **Chain of Thought на старте.** «Сначала опиши план в 5 пунктах, потом пиши код» — резко снижает количество переделок.
2. **Few-shot из своего кода.** Вставьте 1–2 примера в вашем стиле — модель скопирует конвенции (naming, errors, logging).
3. **Reflection loop.** «Покажи код → найди 3 проблемы → исправь». Работает лучше, чем «напиши сразу идеально».
4. **Test-first.** «Сначала тесты, потом реализация». Заставляет модель уточнить контракт.
5. **Decomposition.** Большие задачи режьте на шаги. LLM теряется на «напиши мне приложение», но отлично делает «реализуй эндпоинт /api/users с такой-то схемой».
6. **Show, don't tell.** Дайте ссылку на файл, скриншот ошибки, лог стектрейса. Не пересказывайте — копируйте.
7. **Ограничьте контекст.** Не «вот весь репозиторий», а «вот эти 3 файла + интерфейс модуля X». Меньше шума → точнее ответ.

**Антипаттерны (вы теряете время, если так делаете):**
- «Напиши мне SaaS для X» без декомпозиции.
- Игнорировать ошибки, которые модель явно показывает в комментариях.
- Принимать первый ответ без чтения diff.
- Не давать модели запускать тесты/линтер (если есть агентный режим).
- Менять модель посреди задачи — теряется контекст рассуждений.

### 🔬 Реальные сценарии и боевые примеры (что делать каждый день)

Это не теория, а живой воркфлоу. Каждый сценарий — с готовым шаблоном промпта, который вы можете скопировать и подставить свои данные.

---

#### 1️⃣ Новая фича с нуля (от ТЗ до PR)

**Цепочка:** `План в Claude → схема БД и API → тесты → реализация по модулям → ревью diff → правки → коммит`. На фичу средней сложности уходит 2–4 часа вместо 1–2 дней.

**Шаблон промпта (этап «план»):**

```
Контекст: backend на FastAPI + Postgres + SQLAlchemy 2.0, async.
Задача: добавить эндпоинт /api/orders с фильтрацией по статусу,
сортировкой по дате и пагинацией (cursor-based).
Требования:
- авторизация через JWT (см. app/auth.py — приложу)
- 200 RPS на одну инстанс
- покрытие тестами ≥ 80%

Шаг 1: предложи план из 5–7 пунктов с декомпозицией на коммиты.
Шаг 2: укажи риски и edge-cases.
Шаг 3: жди моего ОК перед началом кода.
```

**Почему работает:** модель не бросается писать код, а сначала согласует архитектуру. Вы экономите 2 итерации переделок.

---

#### 2️⃣ Дебаг продового бага (Sherlock-mode)

**Что даёте Claude:** стектрейс + 2–3 ключевых файла + краткое описание поведения. Просите 3 гипотезы и план диагностики. В 70% случаев он угадывает причину с первого раза.

**Шаблон:**

```
Симптом: после деплоя версии 2.4.1 в проде растёт latency p99
с 200 мс до 1.2 с на эндпоинте /api/search.
CPU и память в норме. Ошибок 5xx нет.

Прикладываю:
- стектрейс из APM (см. ниже)
- diff между 2.4.0 и 2.4.1 (app/search/service.py)
- метрики БД за последние 24 часа

Дай 3 наиболее вероятные гипотезы (от сильной к слабой)
с обоснованием. Для каждой — конкретные команды/запросы для проверки.
Не пиши код, пока не выберем гипотезу.
```

**Pro tip:** если первая гипотеза не подтвердилась — не уходите в новый чат, скажите модели «гипотеза 1 отпала, вот данные эксперимента, обновляй приоритеты».

---

#### 3️⃣ Рефакторинг легаси на 2000+ строк

**Инструмент:** Cursor Composer или Claude Code — потому что нужны мультифайловые правки и удержание контекста.

**Шаблон:**

```
Файл: app/legacy/billing.py (2147 строк, написан в 2019, без тестов).
Цель: разнести на чистые слои (domain / service / repository / adapters),
сохранить публичный API (см. список функций в __all__),
покрыть тестами критичные пути (расчёт, налоги, возвраты).

План работы:
1. Сначала характеризационные тесты на текущее поведение (golden master).
2. Извлечение domain-моделей (dataclasses, без зависимостей).
3. Выделение repository (всё, что ходит в БД).
4. Service-слой (бизнес-логика).
5. Adapters (внешние API: Stripe, налоговая).

Работаем итерациями по 200–300 строк. После каждого шага:
- запускаешь тесты (pytest -x)
- показываешь diff
- ждёшь моего ОК перед следующим шагом.
```

**Ключевой момент:** характеризационные тесты в начале. Без них рефакторинг = русская рулетка.

---

#### 4️⃣ Чтение и понимание чужой кодбазы

**Инструмент:** Gemini 2.x Pro (1M контекст) или Claude Projects с загруженным репо. Альтернатива — Aider `/map` или Cursor `@codebase`.

**Воркфлоу:**

```
1. «Опиши архитектуру проекта в 1 абзаце + диаграмма (Mermaid)».
2. «Где обрабатывается логин пользователя? Покажи путь от роута до БД».
3. «Какие точки расширения есть в модуле X? Где обычно добавляют новые провайдеры?».
4. «Найди мёртвый код: функции, которые нигде не вызываются».
5. «Что сломается, если я переименую User.email в User.contact_email?».
```

За час разбираетесь в проекте, на изучение которого ушла бы неделя.

---

#### 5️⃣ Перевод между языками и фреймворками

**Типовые маршруты:** Python → Go (для перформанса), Express → FastAPI, React (CRA) → Next.js, REST → GraphQL, monolith → микросервисы.

**Шаблон:**

```
Перепиши модуль из Python (FastAPI) в Go (Gin), сохраняя:
- логику обработки заказов (см. orders.py)
- JSON-схему ответов (snake_case → snake_case через теги)
- семантику ошибок (HTTP-коды и тела)

Принципы перевода:
- идиоматичный Go (errors.Is, context, structured logging via slog)
- никаких прямых калек с Python (no exceptions, only error returns)
- генерация типов из OpenAPI, если есть

Покажи план перевода, потом перенос по файлам.
```

**Реальная экономия:** черновик 80% качества за 30 минут вместо 2 дней ручного переписывания.

---

#### 6️⃣ Тесты, доки, миграции (рутина, которую больше не пишем руками)

**Pytest на готовый код:**
```
Покрой тестами файл app/services/payment.py:
- 100% веток (branch coverage)
- параметризованные тесты для всех валидаций
- mock внешних API (Stripe) через respx
- snapshot-тесты для сериализаторов
Используй pytest-asyncio и factory-boy. Стиль — см. tests/conftest.py.
```

**Docstrings и README:**
```
Сгенерируй Google-style docstrings для всех публичных функций
файла X. Также добавь usage-секцию в README с 3 примерами.
```

**Alembic-миграция:**
```
Сгенерируй миграцию: добавить поле phone (varchar 20, nullable)
в users, индекс по phone, бэкфилл из profiles.phone_number
батчами по 10000 строк. Учти, что таблица 50M строк, миграция
должна быть online (без блокировок).
```

---

#### 7️⃣ Архитектурный спарринг

**Когда:** перед началом большой задачи, при выборе стека, при дизайне сложного компонента.

**Шаблон:**

```
Задача: построить real-time ленту уведомлений (web + mobile),
100K активных пользователей, p95 доставки ≤ 2 сек.

Текущий стек: Python, Postgres, Redis, AWS.
Команда: 3 backend, 2 mobile, без отдельных infra.

Предложи 3 архитектурных подхода:
1. Polling + Redis Streams
2. WebSocket + Pub/Sub (Redis или Kafka)
3. Push-сервис (Pusher / Ably / SNS+APNS/FCM)

Для каждого — стоимость, сложность, риски, время на MVP.
Не выбирай за меня, дай таблицу сравнения.
```

**Что НЕ делать:** не принимать первый ответ как истину. LLM — собеседник, а не оракул. Прогоните идею через коллегу.

---

#### 8️⃣ Code review своего PR перед отправкой

```
Сделай ревью моего PR (diff приложен).
Чек-лист:
1. Безопасность (SQL injection, XSS, secrets в коде).
2. Производительность (N+1 запросы, лишние аллокации).
3. Обработка ошибок (что произойдёт при таймауте/5xx от внешнего API?).
4. Тесты (что не покрыто? какие edge-cases пропущены?).
5. Читаемость (имена, длина функций, дублирование).
6. Совместимость API (ломаем ли мы клиентов?).

Выдай результат в формате: 🔴 блокеры / 🟡 нит-пики / 🟢 что хорошо.
```

Запускайте перед отправкой PR — получите ревью на 3–5 минут раньше, чем коллеги.

---

#### 9️⃣ Изучение новой технологии за вечер

```
Я backend-разработчик с 5 годами Python.
Хочу за вечер въехать в Rust на уровне «могу читать чужой код
и писать простой CLI».

План:
1. Карта концептов: что нового по сравнению с Python (ownership, borrowing, lifetimes, traits, enums-as-ADT).
2. 5 ключевых отличий, на которых спотыкаются Python-разработчики.
3. Мини-проект: переписать вот этот Python-скрипт (приложу) в Rust.
4. Чек-лист «понял/не понял» в конце.

Объясняй через аналогии с Python. Без воды.
```

---

#### 🔟 Pet-проект «вечер пятницы» (полный AI-воркфлоу)

```
Хочу за 3 часа сделать Telegram-бота, который:
- принимает голосовое сообщение
- транскрибирует через Whisper
- саммаризирует через Claude
- отправляет краткий текст обратно

Стек: Python + aiogram 3 + httpx.
Деплой: Fly.io (бесплатный tier).

Декомпозируй на 6 шагов по 30 минут. На каждом шаге:
- что сделать
- какой код написать (с тестом «работает / не работает»)
- частые грабли.

В конце — чек-лист продакшен-готовности (логи, ошибки, rate limits, секреты).
```

**Закон вайбкодинга:** если задача укладывается в один вечер с LLM — делайте сегодня, не откладывайте. Это лучший способ нарастить «насмотренность».

---

### ⚠️ Ограничения и подводные камни

- **Галлюцинации API.** Модель уверенно зовёт несуществующие методы. Спасает: давать актуальные доки в контекст, использовать tool use со встроенным веб-поиском.
- **Дрейф стиля.** Без few-shot из вашего кода LLM пишет «среднеинтернетный» код. Лечится загрузкой 1–2 эталонных файлов.
- **Безопасность.** Не отдавайте ключи, токены, PII даже в платных API. Используйте локальные модели (DeepSeek, Qwen, Llama) для чувствительного кода.
- **Каскад ошибок.** Если модель ошиблась в начале — она будет защищать ошибку. Лечится «откатись и подумай заново», новой сессией, сменой модели.
- **Context rot.** На длинных сессиях качество падает. Лечится сжатием контекста, переходом в новую сессию с кратким summary.
- **Юридические риски.** Лицензии генерируемого кода — серая зона. Для коммерческого продукта читайте политики провайдера.
- **Когнитивная атрофия.** Если только промптите и не думаете — навыки деградируют. Раз в неделю пишите что-то руками без LLM.

### 📚 Курсы и материалы по AI-инжинирингу и vibe coding

**Бесплатно (must-have):**
- **[Anthropic — Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)** — официальный курс от создателей Claude. Лучший старт по промптингу.
- **[Anthropic — Building with Claude (docs + cookbook)](https://docs.anthropic.com)** — официальные гайды, рецепты, tool use, агенты.
- **[Anthropic Skilljar Academy](https://anthropic.skilljar.com)** — бесплатные курсы по Claude Code, AI fluency, агентам.
- **[OpenAI Cookbook](https://cookbook.openai.com)** — рабочие примеры по GPT, function calling, RAG, fine-tuning.
- **[DeepLearning.AI — короткие курсы](https://learn.deeplearning.ai)** — десятки 1–2-часовых курсов: ChatGPT Prompt Engineering for Developers, LangChain, Building Agents, MCP и др. Бесплатно.
- **[Hugging Face — Agents Course](https://huggingface.co/learn/agents-course)** — полный курс по AI-агентам, бесплатно с сертификатом.
- **[Cursor Docs + Forum](https://docs.cursor.com)** — практические гайды по работе в Cursor.
- **[Aider — Tips & Best Practices](https://aider.chat/docs/usage/tips.html)** — концентрат опыта по работе с CLI-агентом.

**На русском:**
- **[@ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data)** — регулярные разборы новых релизов и техник.
- **[Хабр — тег «промпт-инжиниринг»](https://habr.com/ru/search/?q=промпт-инжиниринг)** — практические статьи от русскоязычного коммьюнити.
- YouTube-каналы: ищите разборы Claude Code, Cursor, агентных воркфлоу — экосистема растёт быстро.

**Платно (если готовы инвестировать):**
- **[Maven — AI Engineering cohorts](https://maven.com)** — короткие интенсивы от практиков (LLM в проде, RAG, агенты).
- **[Scrimba — AI Engineering Path](https://scrimba.com)** — интерактивные курсы по работе с LLM API.
- Книги: **«AI Engineering»** (Chip Huyen, 2025), **«Designing Machine Learning Systems»** (Chip Huyen), **«Prompt Engineering for LLMs»** (O'Reilly).

### 🏆 Уровни мастерства vibe coding

| Уровень | Что умеет | Сколько времени до него |
|---|---|---|
| 🥉 **Новичок** | Спрашивает ChatGPT функции, копирует код, не читает diff | 1 неделя |
| 🥈 **Уверенный пользователь** | Cursor/Copilot в IDE, структурированные промпты, тесты | 1–2 месяца практики |
| 🥇 **Продвинутый** | Агенты (Claude Code, Aider), мультифайловые правки, RAG над своим кодом | 3–6 месяцев |
| 💎 **Эксперт** | Кастомные агенты, MCP-серверы, fine-tuning под свой стиль, ведёт фичи end-to-end с LLM | 6–12 месяцев |
| 🧙 **Guru** | Строит AI-first продукты, понимает как модель «думает» изнутри, может объяснить ошибки через архитектуру трансформера | 1–2 года + ML база |

### 🎓 Путь к «vibe coding guru»: 90-дневный план

**Дни 1–30 — фундамент:**
- Пройти Anthropic Prompt Engineering Tutorial.
- Поставить Cursor / Claude Code, сделать 3 пет-проекта (CRUD, парсер, бот).
- Прочитать OpenAI Cookbook по function calling.
- Освоить structured prompting (контекст → цель → ограничения → критерий).

**Дни 31–60 — агенты и RAG:**
- Пройти DeepLearning.AI «Building Agents» и Hugging Face Agents Course.
- Сделать RAG-бота над своими заметками/доками (LangChain или LlamaIndex).
- Подключить MCP-сервер к Claude Code, написать свой инструмент.
- Освоить мультифайловые правки в Cursor Composer / Aider.

**Дни 61–90 — прод и глубина:**
- Деплой LLM-приложения в прод (FastAPI + Claude/GPT + векторная БД).
- Eval-framework: измеряйте качество ответов (RAGAS, promptfoo).
- Прочитать «AI Engineering» Chip Huyen.
- Завести технический блог: 1 статья в неделю про свой опыт.
- Изучить базу трансформеров (Karpathy «Let's build GPT» на YouTube) — чтобы понимать, **почему** LLM ведёт себя так, а не иначе.

> 🧙 К концу 90 дней вы умеете делать с LLM то, на что у обычного разработчика уходит неделя — за вечер. Это и есть vibe coding на уровне сениора 2026.


---

## 🧱 Курс с нуля до профи: фундамент, который не устаревает

Это не «пройди 5 видосов и стань ML-инженером». Это содержательный курс на 6 уровней, где каждый блок строится на предыдущем. Цель — чтобы вы не «прошли тему», а **понимали, что и почему делаете**, и могли объяснить любую часть пайплайна на собеседовании или в продовом обсуждении.

Принципы курса:

- **Сначала зачем, потом как.** В начале каждой темы — задача из реальной жизни, которая мотивирует, зачем это вообще нужно.
- **Понимаем руками.** Каждая ключевая идея реализуется с нуля на NumPy/Python, прежде чем брать готовую библиотеку.
- **Никакой магии.** Если не можете объяснить, что происходит внутри — возвращаетесь и разбираете глубже.
- **Сложность растёт постепенно.** Не прыгаем в трансформеры с первой недели. Сначала линейная регрессия, потом всё остальное.
- **Каждый блок = маленький работающий проект.** Не «закрытая тема в голове», а артефакт на GitHub.

---

### 🧩 Уровень 0. Окружение и инструменты разработчика

**Зачем:** до того, как писать ML, нужно научиться писать просто код и держать его в порядке. 80% времени ML-инженера — это не модели, а данные, инфраструктура и инструменты.

**Что осваиваем:**

- **Python 3.12+.** Установка через `pyenv` или `uv`. Виртуальные окружения (`uv venv`, `poetry`). Зачем они нужны и почему `pip install` в системный Python — путь к боли.
- **Терминал и Bash.** Навигация, `grep`, `find`, `xargs`, пайпы. Без этого вы не выживете на сервере и в Colab.
- **Git и GitHub.** Не «закоммитил-запушил», а: ветки, rebase, conflict resolution, PR-флоу, `.gitignore`, что такое HEAD и почему он detached.
- **VS Code / Cursor.** Расширения: Python, Jupyter, GitLens, Ruff, Pylance. Горячие клавиши, multi-cursor, debugger.
- **Jupyter / Colab.** Когда использовать ноутбук, а когда — `.py` файл. Почему «один длинный ноутбук» — антипаттерн.

**Практика (артефакт на GitHub):**

> CLI-утилита `csv-stats`: принимает CSV-файл, выводит статистику по колонкам (типы, пропуски, мин/макс/среднее). С тестами на pytest, README, `pyproject.toml` через uv. Залить на GitHub с workflow на проверку линтером Ruff.

**Когда переходить дальше:** когда можете с нуля развернуть проект, написать функцию с тестами, закоммитить и открыть PR без гугления каждого шага.

---

### 🧩 Уровень 1. Python для работы с данными

**Зачем:** все данные в мире — это таблицы, тексты и тензоры. Сначала учимся жить с таблицами, потому что 90% реальных задач — табличные.

**Что осваиваем:**

#### Python на инженерном уровне

- Структуры данных: `list`, `dict`, `set`, `tuple` — когда что выбирать, сложность операций.
- Comprehensions и генераторы. Почему `(x for x in ...)` ≠ `[x for x in ...]` и когда это критично для памяти.
- ООП: `@dataclass`, наследование, протоколы (`typing.Protocol`).
- Типизация и `mypy`. Почему типы — это документация, которую невозможно «забыть обновить».
- Контекстные менеджеры (`with`), декораторы, `functools`.
- Асинхронность: базовое понимание `async/await` — пригодится для FastAPI и LLM-агентов.

#### NumPy

- Векторизация: почему `a + b` для массивов в 100 раз быстрее цикла Python.
- Broadcasting: правила и интуиция. Без этого ML-код будет в ошибках формы тензоров.
- Индексация: slicing, fancy indexing, boolean masks.
- Линейная алгебра в NumPy: `@`, `np.linalg.solve`, `np.linalg.svd`.

#### Pandas

- DataFrame и Series, индексы, MultiIndex.
- `groupby`, `merge`, `pivot_table`, `apply` vs векторизованные операции.
- Работа с датами, категориями, пропусками.
- Чтение/запись: CSV, Parquet, SQL. Почему Parquet почти всегда лучше CSV для аналитики.

#### Визуализация

- `matplotlib`: понимать слои (Figure, Axes), а не копировать со Stack Overflow.
- `seaborn`: быстрые статистические графики.
- `plotly`: интерактив, когда нужно отдать дашборд клиенту.

**Практика:**

> Полноценный EDA-проект на открытом датасете (например, [Airbnb listings](https://insideairbnb.com)): загрузка → очистка → пропуски → распределения → корреляции → 5 интересных инсайтов в виде графиков → README с выводами. Это репетиция того, чем занимается аналитик в первую неделю на работе.

**Когда переходить дальше:** когда можете без подсказок сделать EDA любого датасета за 2–3 часа и сформулировать 5 содержательных гипотез о данных.

---

### 🧩 Уровень 2. Математика, которая реально нужна

**Зачем:** ML — это прикладная математика. Без неё вы — «оператор sklearn»: знаете, какую кнопку нажать, но не понимаете, что происходит и почему модель ведёт себя странно. Не нужно быть математиком — нужны рабочие интуиции.

#### Линейная алгебра

- Векторы и матрицы как «таблицы данных» и как «преобразования пространства» (две точки зрения, обе важны).
- Скалярное произведение и его геометрический смысл (косинусная близость → embeddings).
- Матричное умножение: что значит `A @ B` интуитивно.
- Собственные числа, SVD, PCA — почему сжатие размерности работает.

> Источник интуиции: [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra). Смотрите целиком, без перемоток. Это часы, которые окупятся в десятки раз.

**Практика:** реализуйте PCA с нуля на NumPy и сожмите MNIST с 784 признаков до 50. Сравните точность kNN до и после.

#### Математический анализ

- Производная как «скорость изменения». Градиент как «направление наискорейшего роста».
- Правило цепочки. Это фундамент обратного распространения ошибки.
- Частные производные и якобиан — для понимания backprop.

**Практика:** реализуйте градиентный спуск с нуля для функции `f(x, y) = x² + 3y² + sin(x)` и нарисуйте траекторию спуска на контурном графике.

#### Теория вероятностей и статистика

- Случайные величины, распределения (нормальное, биномиальное, Пуассона) — где встречаются в жизни.
- Условная вероятность и теорема Байеса. Это база для всей вероятностной классификации.
- Матожидание, дисперсия, ковариация.
- ЦПТ (центральная предельная теорема) — почему «нормальное распределение везде».
- Гипотезы, p-value, доверительные интервалы. Что они **на самом деле** означают (большинство применяет неверно).
- A/B-тесты: t-test, z-test, размер выборки, ошибки I и II рода, мощность.

**Практика:**

> Симуляция A/B-теста: сгенерируйте две выборки с разным средним, посчитайте t-test, постройте графики мощности теста от размера выборки. Это даст понимание, **почему** «давайте подсмотрим результаты через день» — путь к ложным выводам.

**Источники, которые работают:**

- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) — спокойный темп, много упражнений.
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) — простыми словами про сложное, лучший YouTube по статистике.
- Книга: **«Practical Statistics for Data Scientists»** (Peter Bruce, Andrew Bruce) — практичный концентрат без академической воды.

**Когда переходить дальше:** когда можете объяснить теорему Байеса на примере «тест на болезнь с 95% точностью» и не запутаться в условных вероятностях. Когда понимаете, почему градиент — это вектор, а не число.

---

### 🧩 Уровень 3. Классический ML: от линейной регрессии до бустинга

**Зачем:** на табличных данных (а это 70% реальных задач в индустрии) классические алгоритмы бьют нейросети. XGBoost и LightGBM до сих пор выигрывают большую часть Kaggle-соревнований на табличке. Знать классику — значит уметь решать задачи, за которые платят.

#### 3.1. Постановка задачи ML

- Что такое supervised, unsupervised, self-supervised, reinforcement learning. Где границы.
- Регрессия vs классификация. Бинарная vs многоклассовая vs мультилейбл.
- Train / val / test split. Почему «утечка теста в обучение» = смерть модели в проде.

#### 3.2. Линейная регрессия с нуля

Не «`LinearRegression().fit(X, y)`», а пошагово:

1. Формулируем модель: `y = Xw + b`.
2. Выводим функцию потерь: MSE.
3. Считаем градиент аналитически.
4. Реализуем градиентный спуск на NumPy.
5. Сравниваем с `np.linalg.lstsq` и со sklearn.

Это занятие, после которого «нейросети» перестают быть магией. Линейная регрессия — это нейросеть из одного нейрона без активации.

#### 3.3. Логистическая регрессия и классификация

- От линейной к логистической: зачем сигмоида.
- Кросс-энтропия как функция потерь и почему именно она.
- Многоклассовая классификация: softmax.

**Практика:** руками на NumPy обучить логрегрессию различать «кошка/собака» по двум придуманным признакам. Нарисовать decision boundary.

#### 3.4. Метрики и почему accuracy — плохо

- Точность (accuracy), precision, recall, F1, ROC-AUC, PR-AUC.
- Confusion matrix и как её читать.
- **Когда какая метрика лучше:** медицинская диагностика, кредитный скоринг, рекомендации, поиск — для каждого случая своя метрика.
- Калибровка вероятностей: модель может быть точной, но «слишком уверенной».

#### 3.5. Регуляризация и переобучение

- Что такое overfitting и underfitting на интуитивном уровне.
- L1 (Lasso) и L2 (Ridge) — что они делают и когда что выбрать.
- Bias-variance tradeoff — главная картинка ML, должна быть нарисована от руки.
- Кросс-валидация: k-fold, stratified, time series split.

#### 3.6. Feature engineering

- Категориальные признаки: one-hot, target encoding, frequency encoding.
- Численные: масштабирование, log-преобразование, бининг.
- Признаки из дат: день недели, час, праздники, лаги.
- Взаимодействия признаков. Когда они помогают, а когда — overfitting.

> 🎯 Главный секрет классического ML: **в табличных задачах хороший feature engineering часто бьёт сложную модель**. Время, потраченное на признаки, окупается в 10 раз сильнее, чем подбор гиперпараметров.

#### 3.7. Деревья решений и ансамбли

- Дерево решений: как оно строится, что такое gain и impurity.
- Случайный лес: bagging и почему «много слабых = одна сильная».
- Градиентный бустинг: интуиция через «исправление ошибок предыдущей модели».
- XGBoost, LightGBM, CatBoost — чем отличаются и когда что брать.

**Практика на этом уровне:**

> Конкретный проект «от данных до сабмита» на Kaggle ([House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) или [Titanic](https://www.kaggle.com/competitions/titanic)): EDA → feature engineering → 3 модели (Ridge, RandomForest, LightGBM) → кросс-валидация → блендинг → сабмит. Цель: попасть в топ 30%. Описать всё в публичном ноутбуке с объяснениями.

**Источник:** [ODS ML Course (mlcourse.ai)](https://mlcourse.ai) — лучший русско/англоязычный курс по классическому ML с заданиями.

**Когда переходить дальше:** когда можете без подсказок взять любой табличный датасет и за день собрать рабочий ML-пайплайн с честной кросс-валидацией и осмысленным feature engineering.

---

### 🧩 Уровень 4. Нейросети: понимаем изнутри, потом используем фреймворки

**Зачем:** нейросети — это не «другой алгоритм», а обобщение того, что вы уже умеете. Понимание изнутри отличает инженера от «оператора PyTorch».

#### 4.1. Перцептрон и backprop с нуля

- Один нейрон = логистическая регрессия. Слой нейронов = много логрегрессий параллельно.
- Функции активации: ReLU, GELU, sigmoid, tanh — зачем нужна нелинейность.
- Прямой проход (forward): композиция линейных слоёв и активаций.
- Обратное распространение (backprop): правило цепочки в действии.

> Обязательное упражнение: пройдите [Karpathy — Neural Networks: Zero to Hero, лекции 1–2](https://karpathy.ai/zero-to-hero.html). Реализуйте `micrograd` (автодифф на 100 строк) и MLP с нуля. Это переломный момент в обучении.

#### 4.2. Обучение нейросети

- Mini-batch SGD: почему не «один объект» и не «весь датасет».
- Learning rate, его выбор и расписания (cosine, warmup, step).
- Оптимизаторы: SGD with momentum → Adam → AdamW. Что меняется.
- Регуляризация в нейросетях: dropout, weight decay, early stopping, augmentations.
- Batch / Layer / Group normalization — что они на самом деле делают.

#### 4.3. PyTorch как инструмент

- Тензоры, автоград, `nn.Module`, `DataLoader`, `optimizer.step()`.
- Тренировочный цикл: что должно быть внутри, что вынести.
- Перенос на GPU, mixed precision (`autocast`).
- Сохранение/загрузка модели, инференс.

**Практика:**

> Свой классификатор MNIST: сначала MLP, потом простая CNN. Обучить на Colab GPU. Достичь ≥ 99% на тесте. Понять каждую строчку кода.

#### 4.4. CNN: компьютерное зрение

- Свёртки: интуиция через детектор краёв.
- Pooling, padding, stride.
- Архитектуры: LeNet → AlexNet → VGG → ResNet (зачем skip-connections) → EfficientNet.
- Transfer learning: почему обучение с нуля почти никогда не нужно.

**Практика:** классификатор пород собак (или своего датасета) через transfer learning на ResNet или ConvNeXt. Деплой как Hugging Face Space с веб-интерфейсом на Gradio.

#### 4.5. RNN, LSTM, эпоха «до трансформеров»

Это короткий блок, но важный, чтобы понять, **почему трансформеры взлетели**.

- RNN: идея последовательной обработки, проблема исчезающего градиента.
- LSTM/GRU как костыли вокруг RNN.
- Seq2seq и attention — мостик к трансформерам.

**Практика:** простая RNN, генерирующая текст символ-за-символом на корпусе Пушкина или Шекспира. Чтобы прочувствовать, насколько RNN слабее трансформера.

#### 4.6. Трансформеры с нуля

Это **самая важная тема современного ML**. Не пропускайте.

- Self-attention: интуиция через «каждое слово смотрит на каждое».
- Query, Key, Value — что они означают.
- Multi-head attention: почему «много голов» лучше одной.
- Positional encoding: как трансформер понимает порядок.
- Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — для каких задач что.

> Обязательное упражнение: пройдите [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). Реализуйте nanoGPT и обучите на своих данных (свои сообщения в Telegram, тексты Достоевского, что угодно). После этого LLM перестают быть магией.

**Когда переходить дальше:** когда можете на доске нарисовать архитектуру трансформера, объяснить, что происходит на каждом шаге forward pass, и написать его базовую версию на PyTorch за час.

---

### 🧩 Уровень 5. LLM, RAG и прикладной AI

**Зачем:** на 2026 год это самая горячая область с самыми высокими зарплатами. И — что важнее — это рабочий инструмент, который умножает вашу продуктивность как инженера.

#### 5.1. Pre-training, SFT, RLHF, DPO — как обучают современные LLM

- Pre-training: предсказание следующего токена на триллионах токенов интернета.
- SFT (supervised fine-tuning): дообучение на парах «инструкция → ответ».
- RLHF (reinforcement learning from human feedback): почему первые GPT были «умные, но грубые», а ChatGPT стал «вежливым».
- DPO (direct preference optimization): современная альтернатива RLHF.
- Constitutional AI, RLAIF — подход Anthropic.

**Не нужно:** обучать свою LLM с нуля. Это $10M+ и месяцы. **Нужно:** понимать процесс, чтобы знать, что модель умеет и почему.

#### 5.2. Работа с LLM через API

- Anthropic Claude API, OpenAI API, open-source модели через vLLM/TGI.
- Параметры: temperature, top_p, top_k, max_tokens, stop sequences.
- Streaming, function calling, structured outputs (JSON mode).
- Стоимость и оптимизация: кеширование промптов, выбор модели под задачу.

**Практика:** консольный ассистент на Claude API, который умеет читать локальные файлы (через function calling) и отвечать на вопросы по ним.

#### 5.3. Промптинг как навык

- Zero-shot, few-shot, chain-of-thought.
- Декомпозиция сложной задачи на шаги.
- Структурированные промпты (контекст / задача / формат / примеры / критерий готовности).
- Eval промптов: как понять, что новый промпт лучше старого (не «на глаз»).

#### 5.4. Embeddings и векторный поиск

- Что такое embedding-вектор. Косинусная близость как мера смысла.
- Модели: `text-embedding-3`, `bge`, `e5`, `gte` — что выбрать.
- Векторные БД: Qdrant, Pinecone, pgvector, Chroma.
- Метрики поиска: recall@k, MRR, NDCG.

#### 5.5. RAG (Retrieval-Augmented Generation)

- Зачем: LLM не знает ваши данные, у неё ограниченный контекст, и она галлюцинирует.
- Пайплайн: chunking → embedding → vector store → retrieval → reranking → prompt → LLM → answer.
- Reranking: cross-encoder поверх векторного поиска — почему +20% к качеству.
- Eval RAG: RAGAS, promptfoo, golden datasets.

**Практика:**

> Полноценный RAG-сервис по вашим заметкам / документации компании / любым PDF: FastAPI + Qdrant + sentence-transformers + Claude. С метриками качества и Streamlit-фронтом. Это уже коммерчески ценный артефакт.

#### 5.6. Агенты и tool use

- Что такое агент: LLM + цикл «думаю → действую → наблюдаю».
- Tool use / function calling: подключаем LLM к внешним системам.
- MCP (Model Context Protocol) — стандарт от Anthropic для инструментов.
- Multi-agent системы: когда несколько LLM решают задачу совместно.
- Где агенты работают (помощники, автоматизация), а где пока ломаются.

#### 5.7. Fine-tuning, LoRA, PEFT

- Когда дообучать, а когда хватит промптинга и RAG (в 90% случаев — хватит).
- LoRA и QLoRA: дообучение больших моделей на одной GPU.
- Подготовка датасета: чистка, формат, размер.
- Eval после fine-tuning: не забываем, что модель могла «разучиться» делать другое.

**Когда переходить дальше:** когда у вас есть в портфолио рабочий RAG-сервис с измеренным качеством и LoRA-дообучение модели на доменных данных.

---

### 🧩 Уровень 6. MLOps и инженерия в проде

**Зачем:** модель в ноутбуке ≠ модель в проде. Уровень, который отличает «человека, который умеет обучать» от «человека, которому платят senior-зарплату».

#### 6.1. Воспроизводимость и трекинг

- Версионирование кода (Git), данных (DVC), моделей (MLflow Registry).
- Эксперименты: трекинг гиперпараметров, метрик, артефактов.
- Seed everywhere: почему обучение должно давать одинаковый результат при повторе.

#### 6.2. Деплой

- Сериализация модели: pickle (плохо), ONNX, TorchScript, safetensors.
- Сервинг: FastAPI (просто), BentoML / Ray Serve / Triton (прод).
- LLM-сервинг: vLLM, TGI, SGLang — что и когда.
- Контейнеризация (Docker) и оркестрация (Kubernetes на базовом уровне).

#### 6.3. Мониторинг и retraining

- Метрики качества в реальном времени.
- Data drift и concept drift: что это и как ловить (Evidently AI, NannyML).
- Когда переобучать: по расписанию, по триггерам, по деградации метрик.
- Shadow deployments, canary, A/B-тесты моделей.

#### 6.4. Data engineering на минимуме, достаточном для ML

- SQL глубоко: оконные функции, CTE, оптимизация.
- ETL/ELT: dbt, Airflow, Prefect, Dagster.
- Feature store: Feast, Tecton — когда нужен, когда оверкилл.
- Стриминг: Kafka, Flink — на уровне «понимаю архитектуру».

**Практика финального уровня:**

> Мини-MLOps платформа: тренировка модели → MLflow трекинг → DVC версионирование данных → Docker-образ → BentoML сервинг → Evidently мониторинг → CI/CD на GitHub Actions. Один репозиторий, один `make deploy`. Это уже сильный senior-проект.

**Когда вы прошли весь курс:** вы можете взять любую задачу из реального бизнеса, спроектировать end-to-end решение, обучить модель, задеплоить, мониторить и обосновать каждое решение на code review. Это и есть «профи».

---

### 🗓️ Как проходить курс по времени

- **Минимальный темп:** 10 часов в неделю → весь курс за **9–12 месяцев**.
- **Интенсивный темп:** 25 часов в неделю → **5–6 месяцев**.
- **С работой и семьёй:** реалистично **12–18 месяцев**. Это нормально.

Расписание по неделям, материалы и конкретные артефакты для каждого уровня — в следующем разделе («🎓 Авторский курс на 26 недель»). Этот раздел — **что и почему**. Следующий — **когда и в каком порядке**.

---

## 🎓 Авторский курс «ML с нуля до прод-инженера за 6 месяцев»

Это не «ещё одна подборка ссылок», а готовая программа: каждую неделю — конкретная тема, материалы, практическое задание и артефакт в портфолио. Если пройдёте до конца, выйдете с GitHub-репозиторием на 15–20 проектов и реальным пониманием, как ML работает в проде. Темп — 10–15 часов в неделю.

### 📌 Принципы курса

1. **Кода больше, чем теории.** Каждая неделя заканчивается артефактом: ноутбуком, скриптом, мини-сервисом.
2. **Один источник на тему.** Не распыляйтесь: выбираем лучший курс по теме и проходим до конца.
3. **Воспроизводим, а не смотрим.** Любая статья = реализация упрощённой версии руками.
4. **Портфолио с первого дня.** Каждый проект → GitHub с README, требованиями и демо.
5. **Сообщество.** Раз в неделю — пост в техблог или Telegram про то, что узнали.

### 🧭 Дорожная карта (26 недель)

#### 🟢 Модуль 1. Фундамент (недели 1–4)

| Неделя | Тема | Материал | Артефакт |
|---|---|---|---|
| 1 | Python для ML | [CS50P (Harvard)](https://cs50.harvard.edu/python/) | CLI-утилита с тестами на pytest |
| 2 | NumPy, Pandas, визуализация | [Kaggle Learn — Pandas](https://www.kaggle.com/learn/pandas) + [Matplotlib tutorial](https://matplotlib.org/stable/tutorials/) | EDA-ноутбук на датасете Titanic |
| 3 | Математика: линейная алгебра | [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | Реализация PCA с нуля на NumPy |
| 4 | Математика: матан + теорвер + статистика | [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) + [StatQuest](https://www.youtube.com/@statquest) | A/B-тест: симуляция и анализ значимости |

**Контрольный вопрос модуля:** объясните своими словами, что такое градиент, и почему обучение нейросети — это спуск по нему.

---

#### 🟢 Модуль 2. Классический ML (недели 5–8)

| Неделя | Тема | Материал | Артефакт |
|---|---|---|---|
| 5 | Линейные модели, регуляризация | [ODS ML Course (mlcourse.ai)](https://mlcourse.ai) — темы 4–5 | Регрессия на цены недвижимости + интерпретация коэффициентов |
| 6 | Деревья, ансамбли, бустинг | [Andrew Ng — Machine Learning Specialization (Coursera, audit free)](https://www.coursera.org/specializations/machine-learning-introduction) | XGBoost/LightGBM на Kaggle-датасете + feature importance |
| 7 | Кросс-валидация, метрики, переобучение | [mlcourse.ai — тема 6](https://mlcourse.ai) | Pipeline с CV, grid search, ROC-AUC, calibration plot |
| 8 | Первый Kaggle-сабмит | [Kaggle Titanic + House Prices](https://www.kaggle.com/competitions) | Сабмит в топ 30%, разбор фич в блог-посте |

**Контрольный проект:** end-to-end ML-пайплайн (загрузка → препроцессинг → обучение → метрики → сохранение модели) с MLflow для трекинга.

---

#### 🟡 Модуль 3. Deep Learning (недели 9–14)

| Неделя | Тема | Материал | Артефакт |
|---|---|---|---|
| 9 | Нейросети с нуля | [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — части 1–2 | micrograd: автодифф на 100 строк |
| 10 | PyTorch + MLP | [PyTorch Official Tutorials](https://pytorch.org/tutorials/) | Классификатор MNIST с нуля, обучение на GPU (Colab/Kaggle) |
| 11 | CNN и компьютерное зрение | [CS231n (Stanford)](http://cs231n.stanford.edu/) — лекции 1–7 | Классификация CIFAR-10, transfer learning на ResNet |
| 12 | RNN, LSTM, attention | [d2l.ai — главы 9–10](https://d2l.ai) | Языковая модель символ-за-символом на Шекспире |
| 13 | Трансформеры с нуля | [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | nanoGPT-клон, обученный на своих данных |
| 14 | Hugging Face + fine-tuning | [HF NLP Course](https://huggingface.co/learn/nlp-course) | Fine-tune BERT на классификации отзывов (RU) |

**Контрольный проект:** свой image classifier на кастомном датасете с деплоем в Streamlit/Gradio Space на Hugging Face.

---

#### 🟡 Модуль 4. NLP, LLM, RAG (недели 15–19)

| Неделя | Тема | Материал | Артефакт |
|---|---|---|---|
| 15 | NLP-фундамент: tokenization, embeddings | [HF NLP Course — Part 1](https://huggingface.co/learn/nlp-course) | Семантический поиск на FAISS + sentence-transformers |
| 16 | Промптинг и API LLM | [Anthropic Prompt Engineering Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) + [OpenAI Cookbook](https://cookbook.openai.com) | CLI-ассистент на Claude API с function calling |
| 17 | RAG end-to-end | [LangChain — RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) | RAG-бот по вашей базе знаний (LlamaIndex + Qdrant) |
| 18 | Агенты и tool use | [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) | Агент, который умеет искать в интернете и вызывать ваше API |
| 19 | Fine-tuning, LoRA, RLHF (обзор) | [HF PEFT docs](https://huggingface.co/docs/peft) + статьи LoRA, DPO | LoRA-дообучение Llama 3 на доменных данных |

**Контрольный проект:** production-ready RAG-сервис (FastAPI + векторная БД + Claude/GPT + eval-фреймворк типа RAGAS).

---

#### 🔴 Модуль 5. MLOps и прод (недели 20–23)

| Неделя | Тема | Материал | Артефакт |
|---|---|---|---|
| 20 | Воспроизводимость и трекинг | [MLflow docs](https://mlflow.org/docs/latest/index.html) + [DVC](https://dvc.org) | ML-проект с версионированием данных и моделей |
| 21 | Деплой моделей | [BentoML / Ray Serve docs](https://docs.bentoml.org) | Сервинг модели через REST + Docker |
| 22 | Мониторинг и дрейф | [Evidently AI tutorials](https://docs.evidentlyai.com) | Дашборд data drift + алертинг |
| 23 | Feature store, ETL, оркестрация | [Feast](https://docs.feast.dev) + [Prefect/Airflow basics](https://www.prefect.io) | Pipeline обучения с расписанием и feature store |

**Контрольный проект:** мини-MLOps платформа: трекинг + сервинг + мониторинг + CI/CD (GitHub Actions).

---

#### 🔴 Модуль 6. Специализация и портфолио (недели 24–26)

| Неделя | Тема | Что делаем |
|---|---|---|
| 24 | Выбор трека | Углубляетесь в **одно** из: NLP/LLM • CV • RecSys • MLOps • Data Engineering • Research. Читаете 5 свежих статей по теме, разбираете 2 руками. |
| 25 | Капстоун-проект | Делаете один большой проект «как в проде»: ТЗ → данные → модель → API → деплой → мониторинг → README с архитектурной схемой. |
| 26 | Карьерный спринт | Резюме (по конкретному треку), LinkedIn/HH, Hugging Face профиль с моделями/спейсами, технический блог-пост по капстоуну, отклики на 20+ вакансий. |

**Финальный артефакт:** GitHub-профиль с 15–20 проектами + 1 «звёздный» капстоун + 5–10 технических постов. С этим вас зовут на собеседования.

### 📦 Что вы получите на выходе

- **Технический стек:** Python, NumPy/Pandas, scikit-learn, PyTorch, Hugging Face, LangChain/LlamaIndex, FastAPI, Docker, MLflow.
- **Понимание основ:** линейка → деревья → нейросети → трансформеры → LLM → агенты. Без «магии».
- **Прод-навыки:** деплой, мониторинг, версионирование, eval. То, что отличает джуна от мидла.
- **Портфолио:** репозиторий, который можно отправить рекрутёру и не стыдно.
- **Сетка контактов:** комьюнити, конференции, Kaggle-команды, открытые PR в популярные репо.

### ⏱️ Сколько это в часах

- Минимальный темп: **10 часов в неделю × 26 недель = 260 часов**. Это 4 месяца full-time или 6 месяцев с работой.
- Реалистичный темп с работой и семьёй: **12–18 месяцев**. Это нормально. Гонка не нужна.
- Главное правило: **лучше 1 час каждый день, чем 10 часов в субботу**. Регулярность бьёт интенсивность.

### 🧪 Как сдавать самому себе

В конце каждого модуля задайте себе 5 вопросов уровня собеседования и ответьте вслух (или запишите видео). Если плывёте — возвращайтесь и углубляйте. Список вопросов — в разделе «📊 Контрольные вопросы по уровням» выше.

> 🎯 Этот курс — не магия. Это дисциплина + правильные источники + регулярная практика. Если пройдёте до конца — вы уже не «человек, который учит ML», а **ML-инженер**. Дальше — рост по треку, который выбрали.


---

## 💰 Деньги, грейды и рынок труда ML в 2026

Это не «инвестиционный совет», а ориентир по российскому и международному рынку на основе открытых данных (hh.ru, getmatch, levels.fyi, RemoteOK, AI Jobs). Цифры приблизительные и зависят от компании, города и удачи.

### 🏷️ Грейды ML-инженера

| Грейд | Что умеет | Опыт | RU (Москва), нетто/мес | EU/US, gross/год |
|---|---|---|---|---|
| **Junior** | Знает Python, scikit-learn, базовый DL. Может обучить модель на готовом датасете. | 0–1.5 года | 120–220K ₽ | $60–90K |
| **Middle** | Самостоятельно ведёт ML-фичу: данные → модель → API. Знает MLOps на базовом уровне. | 1.5–4 года | 250–450K ₽ | $100–160K |
| **Senior** | Архитектурные решения, инфра, ментор команды. Может объяснить любую часть пайплайна. | 4–7 лет | 450–800K ₽ | $160–280K |
| **Staff / Principal / Research** | Технический лидер направления. Исследования, патенты, статьи на воркшопах. | 7+ лет | 800K+ ₽ | $280–600K+ |

> 💡 В FAANG/Big Tech и AI-лабах (OpenAI, Anthropic, DeepMind, Mistral) senior+ позиции с RSU/equity легко уходят за $400–800K total comp. Это потолок индустрии.

### 🌍 Где искать работу

**Россия / СНГ:**
- [hh.ru](https://hh.ru) — фильтр «Машинное обучение», «Data Science», «ML Engineer».
- [GetMatch](https://getmatch.ru) — IT-вакансии с зарплатами.
- [Хабр Карьера](https://career.habr.com) — техно-ориентированный поиск.
- Telegram: `@datascienceml_jobs`, `@Machinelearning_Jobs`.

**Удалёнка и зарубеж:**
- [LinkedIn](https://linkedin.com) — основной канал, обновляйте профиль.
- [AI Jobs](https://aijobs.net), [ML Jobs](https://ai-jobs.net), [RemoteOK](https://remoteok.com).
- [Wellfound (AngelList)](https://wellfound.com) — стартапы, часто с релокацией.
- [levels.fyi](https://levels.fyi) — калькулятор офферов в крупных компаниях.
- Discord-сообщества: Eleuther AI, Hugging Face, LAION.

### 🧭 Стратегия первого оффера

1. **Начните искать за 2–3 месяца до готовности.** Собеседования сами по себе — это обучение.
2. **Подавайтесь широко.** Норма для джуна — 50–100 откликов до первого оффера. Не парьтесь по отказам.
3. **Используйте referrals.** Откликнуться через знакомого в 5–10 раз эффективнее холодного отклика.
4. **Не ведите переговоры из позиции «лишь бы взяли».** Даже джуну можно поднять оффер на 10–20% грамотным разговором.
5. **Первый оффер ≠ окончательный.** Через год переход = +30–50% к зарплате. Это нормально.

---

## 🎤 Подготовка к собеседованиям ML-инженера

Собес на ML-позицию в 2026 — это 4–6 этапов. Готовьтесь системно, не «методом тыка».

### 📋 Что спрашивают (типовая воронка)

| Этап | Длительность | Что проверяют |
|---|---|---|
| **1. Скрининг с рекрутёром** | 30 мин | Мотивация, английский, ожидания по зарплате |
| **2. Технический скрининг** | 45–60 мин | Python, SQL, базовые алгоритмы, классический ML |
| **3. ML-теория** | 60 мин | Метрики, переобучение, регуляризация, бустинг, нейросети, трансформеры |
| **4. ML System Design** | 60–90 мин | Спроектировать рекомендательную систему / поиск / fraud detection end-to-end |
| **5. Coding (LeetCode-style)** | 60 мин | 1–2 задачи medium на структуры данных, реже на ML-алгоритмы с нуля |
| **6. Behavioral / culture fit** | 30–60 мин | STAR-истории, конфликты, ошибки, лидерство |

### 📚 Ресурсы для подготовки

**ML-теория и кейсы:**
- **[Designing Machine Learning Interviews — Chip Huyen](https://huyenchip.com/ml-interviews-book/)** — бесплатная книга, must-read.
- **[ML Interviews — Khang Pham](https://github.com/khangich/machine-learning-interview)** — топовый GitHub-конспект.
- **[Aman.ai — Distilled ML](https://aman.ai)** — справочник по всем темам ML/DL.

**System Design:**
- **[ML System Design Interview — Ali Aminian (книга)](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127)** — золотой стандарт.
- **[System Design Primer](https://github.com/donnemartin/system-design-primer)** — для общего бэкенд-дизайна.

**Coding:**
- **[LeetCode](https://leetcode.com)** — топ-150 задач, фокус на medium.
- **[NeetCode 150](https://neetcode.io)** — структурированный план.
- **[StrataScratch](https://stratascratch.com)** — SQL и data-задачи из реальных собесов.

**Поведенческие:**
- **[Cracking the PM Interview](https://www.crackingthepminterview.com)** — раздел Behavioral подходит и для ML.
- Метод STAR: **S**ituation → **T**ask → **A**ction → **R**esult. Заведите 8–10 готовых историй из опыта.

### 🔥 50 самых частых ML-вопросов (минимум, который надо знать)

**Классика:**
1. Объясните bias-variance tradeoff.
2. Чем L1 регуляризация отличается от L2? Когда что выбрать?
3. Что такое kross-validation и зачем стратификация?
4. Precision, recall, F1, ROC-AUC, PR-AUC — когда какая метрика лучше?
5. Что делать с дисбалансом классов?
6. Почему градиентный бустинг сильнее случайного леса на табличных данных?
7. Объясните, как работает XGBoost / LightGBM / CatBoost изнутри.
8. Что такое feature leakage и как его избежать?
9. Как выбирать пороги классификации?
10. Чем kNN отличается от kMeans?

**Deep Learning:**
11. Как работает обратное распространение ошибки?
12. Зачем нужны функции активации? Сравните ReLU, GELU, Sigmoid.
13. Что такое batch / layer / group normalization?
14. Объясните vanishing/exploding gradients.
15. Чем оптимизаторы Adam, AdamW, SGD отличаются?
16. Что такое dropout и почему он работает?
17. Как устроена CNN? Зачем pooling?
18. Что такое residual connections и зачем они в ResNet?
19. Объясните внимание (attention) и self-attention.
20. Чем encoder-only, decoder-only и encoder-decoder трансформеры отличаются?

**LLM и современный стек:**
21. Что такое токенизация (BPE, WordPiece, SentencePiece)?
22. Объясните positional encoding (sinusoidal, RoPE, ALiBi).
23. Чем pre-training, SFT, RLHF, DPO отличаются?
24. Что такое LoRA / QLoRA / PEFT?
25. Объясните RAG end-to-end.
26. Что такое hallucinations и как с ними бороться?
27. Какие есть метрики качества для LLM (BLEU, ROUGE, BERTScore, LLM-as-judge)?
28. Что такое function calling / tool use?
29. Объясните, почему KV-cache важен для инференса.
30. Чем quantization (INT8, INT4, GPTQ, AWQ) помогает в проде?

**MLOps и прод:**
31. Что такое data drift, concept drift, как мониторить?
32. Как версионировать модели и данные?
33. Что такое feature store?
34. Объясните A/B-тест для ML-модели.
35. Как раскатывать новую модель безопасно (canary, shadow, champion-challenger)?
36. Что такое online vs batch inference?
37. Как организован CI/CD для ML?
38. Чем отличаются sklearn, ONNX, TorchScript, TensorRT для деплоя?
39. Как масштабировать инференс LLM (vLLM, TGI, SGLang)?
40. Что такое retraining strategy и как её выбрать?

**System Design (типовые задачи):**
41. Спроектируйте поиск Instagram / TikTok.
42. Спроектируйте рекомендации YouTube / Netflix.
43. Спроектируйте систему детекции фрода для платёжной системы.
44. Спроектируйте чатбота поддержки на LLM.
45. Спроектируйте систему модерации контента.
46. Как бы вы построили self-driving perception pipeline?
47. Спроектируйте систему персональных уведомлений.
48. Спроектируйте автодополнение поиска (autocomplete).
49. Спроектируйте систему A/B-тестов для рекомендаций.
50. Спроектируйте on-device модель для распознавания речи на телефоне.

> 🎯 Не зубрите ответы. Тренируйте умение **думать вслух**: формулировать предположения, рисовать схему, обсуждать trade-offs. Это и есть собес.

---

## 🚫 ТОП-15 ошибок, которые убивают карьеру в ML

Это типовые грабли, на которые наступают 90% самоучек. Если избежите хотя бы половину — обгоните всех остальных.

1. **Учить теорию без кода.** Прочитанная статья ≠ понятая статья. Реализуйте упрощённую версию руками.
2. **Менять курсы каждую неделю.** Закончите ОДИН до конца, потом второй. Хаотичность — главный враг.
3. **Не делать портфолио.** Без GitHub с проектами вы для HR — невидимка.
4. **Делать «учебные» проекты (Titanic, MNIST) и считать это портфолио.** Все джуны делают одно и то же. Сделайте что-то на своих данных.
5. **Игнорировать инженерную часть.** Без Git, Docker, FastAPI, SQL вы — «ноутбучный» датасайнтист, на которого не очень-то и большой спрос.
6. **Учить «всё подряд».** ML — океан. Выберите трек после первых 3–6 месяцев и углубляйтесь.
7. **Не читать статьи.** Не успеваете за индустрией = устареваете каждые 6 месяцев. 1 статья в неделю — минимум.
8. **Стесняться спрашивать.** Stack Overflow, Discord, чаты, опен-сорс — там сидят люди, которые готовы помочь.
9. **Откладывать поиск работы «пока не выучу».** Идеального момента не будет. Идите на собесы, даже если страшно.
10. **Не учить английский.** 95% сильного контента на английском. Без B2 потолок резко ниже.
11. **Полагаться только на LLM.** AI ускоряет в разы, но не заменяет понимание. Раз в неделю — кодите без подсказок.
12. **Игнорировать математику.** Без неё вы — «оператор sklearn». Не нужно быть PhD, но базу знать обязательно.
13. **Не вести блог / не публиковать.** Объясняя другим, вы понимаете глубже. Плюс это маркетинг для нанимателей.
14. **Сидеть в одиночку.** Найдите учебную группу, ментора, коллег. В одиночку выгорите за 6 месяцев.
15. **Не отдыхать.** ML — марафон. 60 часов в неделю 3 месяца → выгорание → ноль на год. Лучше 12 часов в неделю год.

> 💀 «Готов выучить ML за 3 месяца, по 4 часа в день, без перерывов» — гарантированный путь к выгоранию. Лучше 1 час в день два года.

---

## ✅ Чек-лист готовности к рынку

Распечатайте и отмечайте. Когда отметили 80%+ — пора идти на собеседования.

**Технические навыки:**
- [ ] Python: ООП, типизация, async, тесты, dependency management (poetry/uv)
- [ ] SQL: оконные функции, CTE, оптимизация запросов
- [ ] Git: branching, rebase, conflict resolution, PR-флоу
- [ ] Linux / Bash: базовая работа в терминале, ssh, скрипты
- [ ] Docker: написать Dockerfile, собрать образ, docker-compose
- [ ] NumPy / Pandas: на уровне «могу преобразовать любые данные»
- [ ] scikit-learn: pipeline, cross-val, grid search, feature engineering
- [ ] PyTorch: написать кастомную модель, обучить на GPU
- [ ] Hugging Face: использовать предобученные модели, fine-tune
- [ ] FastAPI / Flask: написать REST-сервис для инференса

**ML-теория:**
- [ ] Линейная алгебра, матан, теорвер, статистика — базовый уровень
- [ ] Метрики, кросс-валидация, регуляризация
- [ ] Линейные модели, деревья, бустинг
- [ ] CNN, RNN, трансформеры — понимаю архитектуру
- [ ] LLM: pre-training, SFT, RLHF, prompting, RAG, fine-tuning
- [ ] MLOps: трекинг, версионирование, мониторинг, деплой

**Портфолио и репутация:**
- [ ] GitHub с 10+ проектами, README, requirements, тесты
- [ ] 1 «звёздный» end-to-end проект с деплоем и мониторингом
- [ ] Kaggle: 2–3 сабмита, желательно top 30%
- [ ] Hugging Face: профиль с моделями или Spaces
- [ ] LinkedIn: профиль с проектами, навыками, рекомендациями
- [ ] Технический блог: 5+ постов или 1 крепкий разбор
- [ ] 1 контрибьюшен в open-source ML-проект (не обязательно крупный)

**Soft и поиск:**
- [ ] Английский: B2+ (читаю статьи, говорю на собесах)
- [ ] Резюме на 1 страницу, заточенное под ML
- [ ] 8–10 STAR-историй из опыта
- [ ] 50 ответов на типовые ML-вопросы
- [ ] 5 ML System Design кейсов разобрано
- [ ] Кружок коллег / менторов / комьюнити, где можно спросить

> ✅ 25+ галок — джуниор готов. 35+ — мидл. 45+ — сильный мидл / претендент на сениора.

---

## 🤝 Как поддержать проект

Этот roadmap — open-source. Если он вам помог:

- ⭐ Поставьте звезду на GitHub — это сигнал, что материал полезен.
- 🔄 Расшарьте друзьям, в чатах, в соцсетях. Особенно тем, кто только начинает.
- 📝 Откройте Issue с предложением: что добавить, что исправить, какие ссылки протухли.
- 🛠 Сделайте Pull Request: исправили опечатку, нашли крутой курс — добавьте.
- 💬 Напишите в `@csharp_ci` или сообществе ML — расскажите, как roadmap зашёл и что улучшить.

**Принцип проекта:** материал должен быть **бесплатным, актуальным и применимым в проде**. Никаких партнёрок, скрытой рекламы и «купи мой курс за 99 000 ₽». Только то, что реально работает.


---

> Этот roadmap по машинному обучению — карта местности, а не маршрут. Маршрут вы прокладываете сами, исходя из задач, рынка и того, что вас зажигает. Удачи на пути от первого `import numpy` до собственной обученной LLM.
