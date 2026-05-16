# Урок 13. Vision Transformers и мультимодальные модели

> Цель урока: разобрать, как трансформер применили к изображениям (ViT), а потом — как «склеили» зрение и язык в одну модель (CLIP, LLaVA). Это путь к современному computer vision и мультимодальным агентам.

## От CNN к ViT: одна простая идея

CNN обрабатывает изображение **локально**: каждый фильтр смотрит на маленький патч. ViT (Vision Transformer, 2020) предложил радикальное упрощение:

1. Режем картинку на патчи 16×16.
2. Делаем из каждого патча вектор (просто `flatten + Linear`).
3. Прибавляем positional embedding.
4. Прокидываем через **обычный трансформер**, как будто это последовательность токенов.

Никаких свёрток. И это работает. На больших датасетах (>100M картинок) ViT обходит CNN.

## ViT с нуля

```python
import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_patches = (img // patch) ** 2

    def forward(self, x):
        # (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
        return self.proj(x).flatten(2).transpose(1, 2)

class ViT(nn.Module):
    def __init__(self, n_classes=1000, dim=768, depth=12, heads=12, patch=16, img=224):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))   # [CLS] token
```

**`[CLS]` токен.** Специальный обучаемый токен, добавленный в начало последовательности. На его выходе после всех блоков делается классификация. Та же идея, что и в BERT.

## DeiT, Swin, ConvNeXt — что важно знать

- **DeiT (2021)** — ViT, обученный без миллиардов картинок, через distillation от CNN-учителя. Доказал, что ViT можно тренировать на ImageNet-1k.
- **Swin Transformer (2021)** — иерархический ViT с локальным attention внутри окон. Хорош для сегментации и детекции.
- **ConvNeXt (2022)** — «трансформер, но из свёрток». Доказал, что современные CNN могут конкурировать с ViT при том же бюджете.

Сейчас в проде вы скорее всего возьмёте `timm` и `vit_base_patch16_224` или `convnext_base`.

## CLIP: соединяем язык и зрение

**CLIP (Contrastive Language-Image Pretraining, OpenAI, 2021)** — модель, которая выучила общее пространство для картинок и текста.

**Архитектура:**

- Image encoder (ViT или ResNet).
- Text encoder (трансформер).
- Оба проецируют в общее пространство размерности 512-768.

**Обучение.** Дано N пар (картинка, описание). Считаем все попарные косинусные сходства — матрицу N×N. Loss: максимизировать сходство на диагонали, минимизировать вне диагонали (contrastive loss):

$$
\mathcal{L} = -\frac{1}{N} \sum_i \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_j \exp(\text{sim}(I_i, T_j) / \tau)}
$$

`τ` — температура (обучаемая).

**Зачем это нужно.** После обучения CLIP может:

- Делать **zero-shot классификацию**: дайте ему N названий классов, картинку — и он скажет, какому классу она ближе. **Не обучая ничего**.
- Искать картинки по текстовому запросу.
- Работать как «глаза» для мультимодальных LLM (LLaVA, GPT-4V).

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = Image.open("cat.jpg")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
inputs = proc(text=labels, images=img, return_tensors="pt", padding=True)
out = model(**inputs)
probs = out.logits_per_image.softmax(dim=-1)
print({l: p.item() for l, p in zip(labels, probs[0])})
```

## BLIP, LLaVA, мультимодальные LLM

**BLIP (2022)** добавил к CLIP-подобной паре генеративный декодер — модель умеет **описывать картинки** и отвечать на вопросы по ним.

**LLaVA (Large Language and Vision Assistant, 2023)** — наглядная архитектура мультимодальной LLM:

1. **CLIP ViT** кодирует картинку в N патч-эмбеддингов.
2. **MLP-projector** проецирует их в пространство эмбеддингов LLM.
3. **LLM (Llama, Mistral)** получает «токены картинки» + текст вопроса как одну последовательность и генерирует ответ.

Это даёт ChatGPT-подобный интерфейс «опиши, что на картинке», «найди ошибку в схеме», «расшифруй чертёж».

**Современный стек** (2024-2025):

- **Qwen2-VL, InternVL** — открытые конкуренты GPT-4o.
- **Idefics3** от HuggingFace.
- **LLaVA-NeXT** — апдейт классической LLaVA.

## SAM, DINOv2, sticky задачи computer vision

- **SAM (Segment Anything, Meta, 2023)** — универсальная сегментация по клику/боксу/тексту. Сменила парадигму: один файл весов + промптинг вместо обучения модели под каждый класс.
- **DINOv2 (2023)** — self-supervised ViT, дающий очень сильные универсальные эмбеддинги картинок без меток. Используется как backbone для всего подряд.
- **YOLO-World, GroundingDINO** — детекция объектов по тексту, open-vocabulary.

## 8 практических заданий

1. **Свой ViT.** Запустите код выше. Замените CNN из урока 06 на ViT-Tiny на CIFAR-10. Сравните accuracy и время обучения.
2. **Pretrained ViT.** Возьмите `timm.create_model('vit_base_patch16_224', pretrained=True)`, дообучите голову на flowers-recognition. Получите accuracy > 95%.
3. **Анализ attention в ViT.** Визуализируйте attention из последнего слоя на конкретной картинке: какие патчи смотрят друг на друга?
4. **CLIP zero-shot.** На своих 100 картинках сделайте классификацию через CLIP без обучения. Сравните с supervised baseline на тех же данных, но обученным с нуля.
5. **Поиск картинок по тексту.** На 1000 картинок (например, из своего фотоархива) сделайте поиск через CLIP-эмбеддинги. Запрос: «закат на море».
6. **Image captioning.** Возьмите BLIP, опишите 20 картинок. Посчитайте, на скольких описание адекватно.
7. **Мини-LLaVA.** Возьмите CLIP ViT, заморозьте, добавьте линейный projector в эмбеддинг-пространство маленькой LLM (например, TinyLlama). Дообучите projector на парах «картинка + вопрос → ответ».
8. **SAM на своих данных.** Сегментируйте 20 объектов на ваших картинках через SAM по точкам/боксам. Посмотрите, где он ошибается.

## Чек-лист урока

- [ ] Я могу нарисовать архитектуру ViT и объяснить роль `[CLS]` токена.
- [ ] Я понимаю, что такое contrastive loss CLIP и зачем нужна температура.
- [ ] Я делал zero-shot классификацию через CLIP.
- [ ] Я понимаю, как устроен мост от ViT-патчей к LLM-эмбеддингам в LLaVA.
- [ ] Я знаю, что такое SAM и DINOv2 и где их применять.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 12](./12-production.md) · [README курса](./README.md) · ▶︎ [Урок 14 — Diffusion-модели](./14-diffusion.md)
