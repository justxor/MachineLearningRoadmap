# 13. CLIP и мультимодальные модели

CLIP (2021) — революция. Открыл эпоху zero-shot CV и multi-modal моделей.

## CLIP: идея

Два encoder'а (text и image) проектируют в общее embedding-пространство. Обучение через **contrastive loss** на 400M пар (картинка, подпись из интернета):
- Positive: правильная пара картинка-подпись.
- Negative: все остальные пары в батче.

После обучения: одно и то же понятие («собака») имеет близкие embeddings в обоих modalitiях.

## Zero-shot classification через CLIP

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

image = Image.open('cat.jpg')
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits = outputs.logits_per_image  # similarity scores
probs = logits.softmax(dim=1)
```

**Zero-shot:** модель не училась на ваших данных, но классифицирует. Точность часто 60-80% on the wild без fine-tune.

## Prompt engineering для CLIP

Качество zero-shot сильно зависит от текстового промпта. Стандартные подходы:

- "a photo of {class}" — общий baseline.
- Шаблон специфичный домену: "a satellite image of {class}".
- Prompt ensembling: усреднение embeddings нескольких шаблонов.

```python
templates = [
    "a photo of a {}",
    "a high-quality photo of a {}",
    "a close-up of a {}",
]
# Усредняем embeddings по шаблонам для каждого класса
```

## Image-text retrieval

Найти картинки по текстовому запросу или наоборот:

```python
# Pre-compute embeddings всех картинок
image_embeds = model.get_image_features(**image_inputs)
image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

# Запрос
query_text = "a sunset over mountains"
text_embeds = model.get_text_features(**text_inputs)
text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

similarities = image_embeds @ text_embeds.T
top_k = similarities.topk(k=5)
```

**Применения:** поиск по фотобанку, content moderation, e-commerce visual search.

## OpenCLIP, EVA-CLIP, SigLIP

- **OpenCLIP** (LAION): open-source replication, обучен на LAION-5B. Часто лучше OpenAI CLIP.
- **EVA-CLIP:** более качественный vision encoder через MAE pretrain.
- **SigLIP** (Google): sigmoid loss вместо softmax, проще, лучше.
- **DFN-CLIP:** Apple, ещё лучше качество на retrieval.

В 2026 для production обычно лучше OpenCLIP или SigLIP, чем OpenAI CLIP.

## BLIP, BLIP-2

**BLIP:** combined CLIP + caption generation. Понимает картинку И генерирует текст.

**BLIP-2:** Q-Former — лёгкая связка между frozen vision encoder и frozen LLM. Очень дешёво обучать.

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

inputs = processor(image, "Question: what is shown? Answer:", return_tensors='pt')
out = model.generate(**inputs)
```

## Grounded CLIP / Grounding DINO

Расширение CLIP на детекцию: «найди на картинке все объекты, соответствующие тексту». Open-vocabulary detection.

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')

# Text query вместо фиксированных классов
texts = ['cat', 'dog', 'person']
```

Открыло возможность детектить произвольные классы без обучения.

## Use cases CLIP в production

1. **Visual search e-commerce.** Indexing товаров через CLIP embeddings, search by query или by image.
2. **Content moderation.** Zero-shot detection NSFW/violence через текстовые промпты.
3. **Image deduplication / similarity.** Найти похожие картинки.
4. **Auto-tagging.** Сгенерировать теги для фотобанка.
5. **Data filtering.** Отфильтровать релевантные картинки из миллионов unlabeled.

## Дистилляция CLIP в маленькую модель

CLIP-Large ~ 1.4GB. Для mobile/edge — distillation:

- **TinyCLIP:** дистиллированный, в 5-10x меньше при ~95% качества.
- **MobileCLIP:** Apple, оптимизирован для мобильного.

## Fine-tuning CLIP

Если domain отличается (медицина, satellite), можно fine-tune:
- **Full fine-tune:** дорого, может потерять generalization.
- **LoRA на vision tower:** дёшево, сохраняет zero-shot.
- **Linear probing:** обучить только linear head поверх CLIP embeddings.

## Антипаттерны

- **OpenAI CLIP в 2026.** Чаще лучше OpenCLIP или SigLIP.
- **Игнорировать prompt engineering.** Zero-shot accuracy сильно зависит от формулировки.
- **CLIP-Large на mobile.** Используйте дистиллированные версии.
- **Search без normalization.** Cosine similarity требует L2-normalized embeddings.

## Задания

1. Реализовать zero-shot classifier на CIFAR-10 через CLIP. Без обучения сравнить с fine-tuned ResNet.
2. Построить image search engine для своей коллекции фоток (1000+). Query by text.
3. Сравнить OpenAI CLIP, OpenCLIP ViT-L и SigLIP на zero-shot ImageNet.
4. Использовать BLIP-2 для image captioning. Качество vs скорость.
5. Использовать Grounding DINO для open-vocabulary detection: «найди мне рюкзаки на фото».
6. Fine-tune CLIP через LoRA на satellite imagery. Замерить gain.

## Чек-лист

- [ ] Понимаю архитектуру и обучение CLIP.
- [ ] Умею делать zero-shot classification и retrieval.
- [ ] Знаю про OpenCLIP, SigLIP, BLIP-2.
- [ ] Понимаю Grounding DINO для open-vocab detection.

## Дальше

➡️ [14-vlm.md](./14-vlm.md) — VLM: vision-language models.
