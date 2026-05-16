# Урок 14. Diffusion-модели: от DDPM до Stable Diffusion

> Цель урока: понять, почему сегодня все генеративные модели изображений (Midjourney, Stable Diffusion, DALL-E) — это диффузия. Реализовать DDPM с нуля на маленьких картинках.

## Идея диффузии в одном предложении

Берём картинку, постепенно зашумляем её до чистого гауссова шума. Учим нейросеть **отменять** этот шум шаг за шагом. На инференсе — стартуем с чистого шума и итеративно убираем его, получая новую картинку.

## Forward process: как испортить картинку

Прямой процесс — это марковская цепь, добавляющая гауссов шум на каждом шаге `t = 1, ..., T`:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

`β_t` — расписание шума (обычно от `1e-4` до `0.02`). После `T ≈ 1000` шагов от исходной картинки не остаётся ничего, кроме шума.

**Хорошая новость:** можно получить `x_t` за один шаг, не итерируя:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

где `α_t = 1 - β_t`, `ᾱ_t = ∏ α_s`. Это позволяет дешево семплировать на любом шаге при обучении.

## Reverse process: учим сеть убирать шум

Хотим обучить `ε_θ(x_t, t)` — нейросеть, которая по зашумлённой картинке и номеру шага предсказывает добавленный шум. Loss — простой MSE (Ho et al., DDPM, 2020):

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\|^2 \right]
$$

Это эквивалентно (с точностью до константы) оптимизации вариационной нижней оценки правдоподобия — но писать так проще.

## Семплинг: возвращаем картинку из шума

Шаг семплинга DDPM:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

Итеративно от `t=T` до `t=1`. На выходе — `x_0`, новая сгенерированная картинка.

**DDIM (2020)** — детерминистический сэмплер: `σ_t = 0`, можно делать большие шаги. Вместо 1000 шагов хватает 20-50.

## Минимальный DDPM с нуля

```python
import torch, torch.nn as nn, torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, eps=None):
    if eps is None: eps = torch.randn_like(x0)
    a = alpha_bar[t][:, None, None, None].to(x0.device)
    return (a.sqrt() * x0 + (1 - a).sqrt() * eps), eps

class TinyUNet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.time_embed = nn.Embedding(T, dim)
        self.enc1 = nn.Conv2d(1, dim, 3, padding=1)
        self.enc2 = nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(dim*2, dim*2, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(dim*2, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        emb = self.time_embed(t)[:, :, None, None]
        h1 = self.act(self.enc1(x))
        h2 = self.act(self.enc2(h1 + emb[:, :h1.size(1)]))
        h = self.act(self.mid(h2))
        h = self.act(self.dec2(h))
        return self.dec1(torch.cat([h, h1], dim=1))

# Обучение на MNIST
model = TinyUNet()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
for step in range(20_000):
    x0 = get_mnist_batch()              # (B, 1, 28, 28), нормировано в [-1, 1]
    t = torch.randint(0, T, (x0.size(0),))
    x_t, eps = q_sample(x0, t)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
```

После 20k шагов сеть уже умеет генерировать узнаваемые цифры из шума.

## Latent Diffusion: как сделать его быстрым (Stable Diffusion)

Прямая диффузия на 512×512 RGB — это 786 432 размерных шагов. Дорого.

**Latent Diffusion (Rombach et al., 2022)** придумал гениальный трюк:

1. Обучаем VAE-энкодер: 512×512×3 → 64×64×4 (в 48 раз меньше пикселей).
2. Делаем диффузию **в этом сжатом пространстве**.
3. На выходе декодер VAE возвращает картинку.

Это Stable Diffusion. Те же самые DDPM-формулы, просто в `latent` пространстве.

## Conditioning: как «направлять» генерацию

**Class-conditional.** Подаём метку класса как embedding, складываем с time embedding.

**Text-conditional.** Текст пропускаем через CLIP text encoder, получаем последовательность эмбеддингов. Добавляем в UNet через cross-attention слои.

**Classifier-Free Guidance (CFG).** Обучаем модель и с условием, и без (с пустым промптом). На инференсе берём:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

`s` — guidance scale, обычно 5-12. Чем выше — тем точнее модель следует промпту, но тем меньше разнообразие.

## DiT и эра трансформеров для генерации

**DiT (Diffusion Transformer, 2023)** заменил UNet на ViT-подобный трансформер. Лучше масштабируется. На этом построены **Sora** (OpenAI) и **Stable Diffusion 3**.

## Beyond images: video, audio, 3D, molecules

- **Stable Video Diffusion, Sora** — диффузия применяется к видео-латентам.
- **AudioLDM, MusicGen** — генерация музыки и звука.
- **Diffusion Policy** — диффузионные модели в робототехнике.
- **AlphaFold-3, RFdiffusion** — диффузия в задачах структурной биологии.

## 8 практических заданий

1. **DDPM на MNIST.** Запустите код выше. Сгенерируйте 16 цифр через DDPM-семплинг (1000 шагов). Сохраните сетку.
2. **DDIM-семплинг.** Реализуйте DDIM-семплер. Сгенерируйте картинки за 20 шагов вместо 1000. Сравните качество.
3. **Расписание β.** Сравните linear и cosine schedule (Nichol & Dhariwal, 2021) на MNIST. Какое даёт лучшее FID?
4. **Class-conditional.** Сделайте условную диффузию: добавьте метку класса в TinyUNet. Сгенерируйте «именно 7» или «именно 3».
5. **Stable Diffusion через diffusers.** Запустите `stable-diffusion-v1-5` через библиотеку `diffusers`. Сгенерируйте 5 картинок по своим промптам.
6. **CFG sweep.** Для одного промпта сгенерируйте картинки с `guidance_scale = 1, 3, 7, 15, 30`. Опишите визуальную разницу.
7. **Свой LoRA для SD.** Возьмите 10-30 своих картинок одного объекта (свой кот, любимая чашка). Обучите LoRA-адаптер через `diffusers` или `kohya_ss`. Сгенерируйте «мой кот в космосе».
8. **ControlNet.** Используйте `ControlNet` с canny-edges: дайте на вход скетч, получите детальную картинку, повторяющую структуру.

## Чек-лист урока

- [ ] Я могу руками вывести формулу `q(x_t | x_0)`.
- [ ] Я понимаю, что предсказывает сеть в DDPM (шум `ε`, а не картинку).
- [ ] Я обучил DDPM с нуля и сгенерировал что-то узнаваемое.
- [ ] Я понимаю, почему latent diffusion работает на 512×512, а пиксельная — нет.
- [ ] Я знаю, что такое CFG и как им регулировать «строгость» к промпту.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 13](./13-vision-transformers.md) · [README курса](./README.md) · ▶︎ [Урок 15 — RLHF и alignment](./15-rlhf-dpo.md)
