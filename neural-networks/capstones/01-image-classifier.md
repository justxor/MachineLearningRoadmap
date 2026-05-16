# Капстон 1. Своя классификация изображений end-to-end

> Делается **после уроков 1-6**. Цель: пройти полный путь от сырого датасета до работающего инференса с веб-демо. Это самый частый «начальный» ML-проект на работе.

## Что должно быть на выходе

- 🗂️ GitHub-репозиторий, например `my-image-classifier`.
- 📊 Обученная CNN, accuracy > 90% на test.
- 📈 README с графиками train/val loss и confusion matrix.
- 🚀 ONNX-модель и скрипт инференса (1 картинка → 1 предсказание).
- 🌐 Веб-демо (Streamlit или Gradio).

## Выбор датасета

**Лучше всего:** придумайте свою тематику. Это **сильно** улучшает резюме.

Примеры (выберите один или придумайте свой):

- **Распознавание пород собак** — 10 пород по 200 фото из Google Images или Kaggle.
- **Классификация поз йоги** — 5-10 поз, фото сами или из открытых датасетов.
- **Свежие vs гнилые фрукты** — фото на телефон в магазине + дома.
- **Чек или не чек** — для бухгалтерского ML-приложения.
- **Дефекты на платах / на тканях / на металле** — industrial CV, открытые датасеты на Kaggle.

Не берите CIFAR-10 — на собеседовании это выглядит как «прошёл туториал».

## Стек

- **PyTorch + torchvision** — модель, аугментации, тренировка.
- **albumentations** — продвинутые аугментации (рекомендуется, не обязательно).
- **Weights & Biases или TensorBoard** — логирование экспериментов.
- **ONNX Runtime** — инференс.
- **Streamlit** или **Gradio** — демо.

## Структура репозитория

```
my-image-classifier/
├── README.md
├── requirements.txt
├── data/                   # .gitignore, ссылка на датасет в README
├── src/
│   ├── dataset.py          # Dataset класс, аугментации
│   ├── model.py            # ResNet18 + кастомная голова или своя CNN
│   ├── train.py            # train loop, EarlyStopping, чекпоинты
│   ├── eval.py             # confusion matrix, classification_report
│   └── export_onnx.py      # экспорт лучшего чекпоинта в ONNX
├── inference.py            # CLI: python inference.py path/to/image.jpg
├── app.py                  # Streamlit demo
└── notebooks/
    └── 01-eda.ipynb        # exploratory: размеры, баланс классов, примеры
```

## Пошаговый план (5-7 дней)

**День 1. Данные.**

- Соберите/скачайте 1000-3000 изображений с метками классов.
- Разделите на train/val/test 70/15/15 **по содержанию**, а не по дате (важно).
- Проверьте баланс классов, размеры, дубли.
- Сделайте `notebooks/01-eda.ipynb` — 5-10 ячеек с разведочным анализом.

**День 2. Baseline.**

- ResNet18 pretrained на ImageNet, замените последний слой на ваши классы.
- Аугментации: `RandomHorizontalFlip + RandomResizedCrop + ColorJitter + Normalize`.
- AdamW lr=1e-3 для головы, lr=1e-4 для backbone (discriminative fine-tuning).
- Обучите 10-20 эпох, замерьте baseline accuracy.

**День 3. Эксперименты.**

- Попробуйте: бóльшую модель (ResNet50, EfficientNet-B0).
- Усиленные аугментации (RandAugment, MixUp).
- Cosine schedule + warmup.
- Зафиксируйте лучший результат, сохраните `best.pt`.

**День 4. Анализ ошибок.**

- Постройте confusion matrix.
- Покажите 20 worst-misclassified примеров. Найдите паттерн ошибок.
- Если есть систематика — исправьте: добавьте данных, аугментацию или измените архитектуру.

**День 5. Production.**

- Экспортируйте лучшую модель в ONNX.
- Напишите `inference.py` — принимает путь к картинке, печатает предсказание.
- Замерьте latency: PyTorch fp32 vs ONNX Runtime fp32 vs ONNX Runtime int8.

**День 6. Демо.**

- Streamlit-приложение: загрузка картинки → предсказание + top-3 классы с вероятностями.
- Деплой на Streamlit Cloud или Hugging Face Spaces (бесплатно).

**День 7. README.**

- 2 страницы текста. Структура: задача → данные → модель → метрики → как запустить → ограничения.
- 3-5 картинок: примеры классов, confusion matrix, train/val loss, скриншот демо.
- Ссылка на live-демо.

## Чек-лист готовности к собеседованию

- [ ] В README понятно с первого взгляда, что это за проект.
- [ ] Указаны конкретные метрики на test, а не «обучил».
- [ ] Есть section «Limitations» — где модель ломается и почему.
- [ ] Чужой человек может запустить ваш код за < 10 минут.
- [ ] Демо работает и не падает на первой же картинке вне распределения.
- [ ] Закоммитите EDA-notebook, train/val кривые в `reports/`.

## Полезные ссылки

- [timm](https://github.com/huggingface/pytorch-image-models) — все современные CNN/ViT в одной либе.
- [albumentations](https://albumentations.ai/) — продвинутые аугментации.
- [Weights & Biases free tier](https://wandb.ai/) — логирование экспериментов.
- [Streamlit](https://streamlit.io/), [Hugging Face Spaces](https://huggingface.co/spaces) — хостинг демо.

---

[← Все капстоны](./README.md) · [README курса](../README.md)
