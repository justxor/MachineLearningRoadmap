# Урок 17. Self-supervised и contrastive learning

> Цель урока: научиться обучать модели **без размеченных данных**. Это главный приём современного DL: pretraining LLM, ViT, DINOv2, CLIP — всё это self-supervised.

## Зачем

Разметить миллион картинок — миллион долларов и год работы. А неразмеченных картинок в интернете — миллиарды, и они бесплатны. Self-supervised learning (SSL) использует **сами данные** как источник сигнала: модель учится решать задачу, которую можно поставить **автоматически**.

## Главные парадигмы SSL

**1. Pretext task — придумываем искусственную задачу.**

- Предсказать следующее слово (LLM).
- Дополнить пропущенные слова (BERT, masked language modeling).
- Восстановить замаскированные патчи картинки (MAE).
- Угадать поворот картинки (RotNet).

**2. Contrastive learning — учим, что «похожее близко, разное далеко».**

- SimCLR, MoCo для картинок.
- CLIP для пар «картинка-текст».

**3. Self-distillation — модель учится у своей же exponential moving average копии.**

- BYOL, DINO, DINOv2.

## SimCLR: contrastive learning картинок (2020)

Идея: к каждой картинке применяем **две разные аугментации** → получаем «положительную пару» `(x_i, x_j)`. Все остальные картинки в батче — отрицательные. Минимизируем NT-Xent loss:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

После pretraining encoder даёт **универсальные эмбеддинги** — на них можно поставить линейный классификатор и получить качество, близкое к supervised. Часто это и есть **главная цель** SSL.

## Минимальный SimCLR

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T

augment = T.Compose([
    T.RandomResizedCrop(96),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=9),
])

class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.enc = encoder              # backbone, например ResNet без head
        h = encoder.feature_dim
        self.proj = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, proj_dim),
        )

    def forward(self, x):
        h = self.enc(x)
        return F.normalize(self.proj(h), dim=-1)

def nt_xent(z1, z2, tau=0.5):
    z = torch.cat([z1, z2], dim=0)             # 2B x d
    sim = (z @ z.T) / tau
    mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    B = z1.size(0)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)

# Тренировочный цикл
for x, _ in loader:
    x1, x2 = augment(x), augment(x)
    z1, z2 = model(x1), model(x2)
    loss = nt_xent(z1, z2)
    loss.backward(); opt.step(); opt.zero_grad()
```

**Критически важно:** агрессивные аугментации, большой батч (минимум 256, лучше 4096+), температура `τ ≈ 0.1-0.5`. На маленьких батчах SimCLR работает плохо — нужны более изощрённые методы.

## MAE: Masked Autoencoder для картинок (2022)

He et al. предложили перенести идею BERT на картинки:

1. Делим картинку на патчи 16×16.
2. **Маскируем 75%** патчей (агрессивная маска — это ключ).
3. Encoder (ViT) видит только видимые 25%.
4. Decoder (маленький ViT) пытается восстановить замаскированные патчи в пиксельном пространстве.
5. Loss — просто MSE по замаскированным пикселям.

**Почему 75%?** В тексте 15% маски достаточно — текст плотный. Картинка избыточна, и при 15% задача тривиальна (соседние пиксели похожи). 75% делает задачу содержательной.

После pretraining decoder выбрасывается, encoder используется как backbone.

## DINOv2: self-distillation без меток (2023)

Идея:

- Две сети: **student** (обучаем) и **teacher** (EMA от student, не обучаем напрямую).
- Картинку аугментируем двумя способами — получаем разные кропы.
- Teacher видит большие глобальные кропы, student — мелкие локальные.
- Student учится предсказывать выход teacher.

Без явных меток модель учится семантически осмысленным эмбеддингам. DINOv2 на 142M картинок даёт фичи, на которых линейная пробка матчит supervised ResNet-50.

## BERT и MLM: тот же приём в NLP

**Masked Language Modeling (MLM)** — родоначальник SSL в NLP:

- В предложении случайно маскируется 15% токенов.
- Модель учится их предсказывать.
- Контекст с обеих сторон → bidirectional.

Это парадигма BERT. GPT использует другую — **autoregressive LM** (предсказать следующий токен), но обе — self-supervised.

## Современные подходы в audio и видео

- **wav2vec 2.0, HuBERT** — contrastive + MLM для аудио. Pretrain без расшифровок → fine-tune на маленькой разметке для ASR.
- **VideoMAE** — MAE, перенесённый на видео-кубы.
- **VJEPA, V-JEPA 2** — предсказание скрытых представлений будущих кадров вместо пиксельной реконструкции.

## SSL для табличных данных

- **TabPFN, TabTransformer** — pretrain на синтетических задачах, zero-shot на табличках.
- **SCARF** — contrastive learning для табличных данных через corrupting фичей.

## 8 практических заданий

1. **SimCLR с нуля.** Реализуйте код выше. Обучите SimCLR на STL-10 (без меток!). Линейная пробка на эмбеддингах должна давать accuracy > 80%.
2. **Влияние аугментаций.** Поочерёдно уберите ColorJitter / GaussianBlur / Crop. Покажите, что качество эмбеддингов падает. Какая аугментация **самая важная**?
3. **Большой батч важен.** Обучите SimCLR с батчем 32, 128, 512. Сравните линейную пробку.
4. **MAE с нуля.** Реализуйте простой MAE: ViT-encoder + лёгкий ViT-decoder + маска 75%. Обучите на CIFAR-10 без меток.
5. **Линейная пробка vs fine-tune.** На обученном MAE-энкодере сравните: (a) frozen + линейный head, (b) full fine-tune. На каком количестве меток разница исчезает?
6. **DINOv2 готовый.** Возьмите `facebook/dinov2-base`, прогоните 1000 своих картинок через него. Сделайте t-SNE — должны увидеть осмысленные кластеры.
7. **BERT MLM с нуля.** Возьмите свой мини-трансформер из урока 09, переделайте loss на MLM (15% токенов заменяем на `[MASK]`). Pretrain на корпусе русского текста.
8. **SSL для табличек.** На любом табличном датасете (например, Adult Income) попробуйте SCARF-pretraining. Сравните с XGBoost.

## Чек-лист урока

- [ ] Я могу объяснить разницу между pretext tasks, contrastive и self-distillation.
- [ ] Я понимаю, почему SimCLR требует большого батча.
- [ ] Я обучил SimCLR или MAE без размеченных данных и получил полезные эмбеддинги.
- [ ] Я понимаю, почему MAE маскирует 75%, а BERT — только 15%.
- [ ] Я знаю, где применять DINOv2 как универсальный image encoder.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 16](./16-gnn.md) · [README курса](./README.md) · ▶︎ [Урок 18 — Распределённое обучение](./18-distributed-training.md)
