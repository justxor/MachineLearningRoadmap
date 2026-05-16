# 📋 Cheatsheet курса — все формулы и приёмы на одной странице

Краткий справочник по всем 18 урокам. Открывайте, когда забыли формулу, гиперпараметр или название слоя.

---

## 1. Базовые операции

**Линейный слой.** `y = Wx + b`. Для батча: `Y = XW^T + b` (PyTorch convention).

**Сигмоида.** `σ(z) = 1/(1+e^-z)`, `σ'(z) = σ(z)(1-σ(z))`.

**Tanh.** `tanh(z) = (e^z - e^-z)/(e^z + e^-z)`, `tanh'(z) = 1 - tanh²(z)`.

**ReLU.** `max(0, z)`. Производная: 1 если z>0, 0 иначе.

**GELU.** `z · Φ(z)` (Φ — CDF нормального). Стандарт в трансформерах.

**Softmax (стабильный).** `exp(z - max(z)) / sum(exp(z - max(z)))`.

**Cross-entropy.** `L = -Σ y_true · log(y_pred)`. Для бинарной — BCE.

## 2. Инициализация

- Tanh / Sigmoid → **Xavier**: `W ~ N(0, 2/(n_in + n_out))`.
- ReLU / GELU → **He / Kaiming**: `W ~ N(0, 2/n_in)`.
- LSTM forget bias → инициализируйте в 1 (помогает сходимости).
- LoRA: `A` — малый rand, `B` — нули (старт с нулевого Δ).

## 3. Backprop в одну строку

Forward: `z = Wx + b; h = σ(z); L = loss(h, y)`.

Backward:
- `dL/dh = ...` (зависит от loss)
- `dL/dz = dL/dh · σ'(z)`
- `dL/dW = dL/dz · x^T`
- `dL/db = dL/dz`
- `dL/dx = W^T · dL/dz`

Проверка градиентов: `(L(W+ε) - L(W-ε)) / (2ε)` должно совпадать с аналитическим до 1e-5.

## 4. Регуляризация

| Приём | Когда |
|---|---|
| **weight_decay** | Всегда. AdamW `1e-4` для CNN, `0.01-0.1` для трансформеров. |
| **Dropout** | Глубокий MLP, RNN. p = 0.1-0.5. Перед последним слоем не ставить. |
| **BatchNorm** | CNN. Требует батч ≥ 16. |
| **LayerNorm** | Трансформеры, RNN. Не зависит от батча. |
| **Early stopping** | Всегда дёшево, всегда полезно. patience = 5-10. |
| **Аугментация** | Картинки: `RandomCrop + Flip + ColorJitter`. Текст: dropout, back-translation. |

## 5. Оптимизаторы

| Оптимизатор | Когда | Дефолт lr |
|---|---|---|
| SGD + Momentum 0.9 | CNN классика (ResNet, ConvNeXt) | 0.1 (cosine schedule) |
| Adam | Sparse-данные, RL | 1e-3 |
| **AdamW** | Всё остальное по умолчанию | 3e-4 (transformer), 1e-3 (CNN) |
| AdamW + warmup + cosine | Pretraining LLM | 3e-4 пик |

**LR Finder:** растите lr экспоненциально 1-2 эпохи, ищите точку перед взрывом loss.

**Gradient clipping:** `clip_grad_norm_(params, 1.0)` для RNN и трансформеров.

## 6. Размеры тензоров — частые формулы

**Conv2d output:** `H_out = floor((H_in + 2·pad - kernel) / stride) + 1`.

**Receptive field после L слоёв 3×3:** `1 + 2·L` (без stride/pool).

**Attention complexity:** время и память `O(T² · d)` по длине последовательности T.

## 7. Attention за 30 секунд

```
scores = Q @ K^T / sqrt(d_k)
[если causal:] scores[mask] = -inf
weights = softmax(scores)
output = weights @ V
```

**Multi-head:** разбить `d_model` на `n_heads` независимых attention, потом concat + Linear.

## 8. Трансформер-блок (Pre-LN)

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

MLP: `Linear(d, 4d) → GELU → Linear(4d, d)`.

**Positional:** learned (GPT-2), sinusoidal (оригинал), RoPE (LLaMA, Mistral).

## 9. Токенизация

- **BPE / WordPiece / SentencePiece** — современный стандарт.
- 1 английское слово ≈ 1.3 токена. Русское ≈ 2-3 токена в GPT-2/4.
- Для русского лучше `xlm-roberta`, `bge-m3` или модели с большим словарём.

## 10. Fine-tuning

| Сколько данных | Метод |
|---|---|
| 100-1k | Feature extraction (frozen backbone) |
| 1k-10k | LoRA, r=8-16 |
| 10k-100k | LoRA, r=32-64 или full fine-tune с малым lr |
| 100k+ | Full fine-tune |

**LoRA:** `ΔW = BA`, обычно target_modules = `q_proj, v_proj`. lr `1e-4`, weight_decay 0.

**QLoRA:** LoRA + 4-bit квантизация базовой модели. Mistral-7B fine-tune на 1× T4.

## 11. RLHF / DPO

**Pipeline:** Pretrain → SFT → Preference learning (DPO/PPO).

**DPO-loss:**
```
L = -log σ(β · ((log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))))
```

β = 0.1-0.5. Чем больше β — тем сильнее модель тянется к выбранным ответам.

**Когда что:**
- Есть пары «лучше/хуже» → DPO.
- Есть бинарные «нравится/нет» → KTO.
- Reasoning (math/code) и проверяемая награда → GRPO.

## 12. Diffusion

**Forward (зашумление):** `x_t = sqrt(ᾱ_t) · x_0 + sqrt(1-ᾱ_t) · ε`.

**Loss:** `MSE(ε, ε_θ(x_t, t))` — сеть предсказывает шум.

**Семплинг (DDIM, быстрый):** 20-50 шагов вместо 1000.

**CFG:** `ε̃ = ε(x, t, ∅) + s · (ε(x, t, c) - ε(x, t, ∅))`, s = 5-12.

**Latent diffusion:** диффузия в `VAE-латенте` 64×64×4 вместо 512×512×3. Это Stable Diffusion.

## 13. SSL / Contrastive

**SimCLR loss (NT-Xent):**
```
L = -log[exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ)]
```

τ = 0.1-0.5. Большой батч (≥256) обязателен.

**MAE:** маскируйте **75%** патчей картинки, восстанавливайте пиксели MSE.

**Готовые универсальные эмбеддинги картинок:** DINOv2.

## 14. GNN (один слой)

**GCN:** `H' = σ(D̃^(-1/2) · Ã · D̃^(-1/2) · H · W)`, где `Ã = A + I`.

**GAT:** агрегация с обучаемыми весами внимания между соседями.

**Уровни задач:** node-level (Cora) / edge-level (link prediction) / graph-level (molecule properties + mean pool).

**Грабли:** over-smoothing уже на 4-6 слоях. Глубокие GNN обычно не работают.

## 15. Распределённое обучение

| Тип | Что шардируется | Когда |
|---|---|---|
| **DDP** | Только батч | Модель влезает в одну GPU |
| **FSDP / ZeRO-3** | Веса + градиенты + opt-states | Не влезает |
| **TP** | Сами матрицы весов | Огромные слои (Megatron) |
| **PP** | Слои по разным GPU | Очень длинные модели |

**Mixed precision:** bf16 на A100/H100, fp16 + GradScaler на старых GPU.

**Gradient accumulation:** эффективный батч = micro_batch × accum × world_size.

**Gradient checkpointing:** память ↓ в √N, время ↑ ~30%.

## 16. Производительность инференса

| Приём | Эффект |
|---|---|
| ONNX Runtime | 2-5× speedup на CPU |
| INT8 quantization | 4× память ↓, +1-3% latency, -0-1% accuracy |
| INT4 (LLM) | 8× память ↓, -1-3% accuracy |
| KV-cache | O(N²) → O(N) для LLM-генерации |
| vLLM PagedAttention | в 10-20× throughput на параллельных запросах |

## 17. Дефолтные гиперпараметры (стартовые точки)

**CNN на CIFAR-10:**
- AdamW lr=1e-3, wd=5e-4, CosineAnnealing 50 эпох.
- batch=128, RandomCrop(32, pad=4) + Flip.

**Pretrain трансформера:**
- AdamW lr=3e-4, β=(0.9, 0.95), wd=0.1.
- Warmup 1-2k шагов → cosine до 10% от пика.
- bf16, gradient clipping 1.0.

**Fine-tune LLM:**
- AdamW lr=2e-5 (full) или lr=1e-4 (LoRA), 1-3 эпохи.

## 18. Главные «когда что не работает»

- **Loss = NaN** → lr слишком большой, fp16 без loss scaling, деление на 0 в `log` (добавьте `+ 1e-9`).
- **Train loss падает, val растёт** → переобучение, см. блок «Регуляризация».
- **Loss застрял на случайном значении** → lr слишком маленький, или сеть инициализирована плохо, или ReLU-мёртвые нейроны.
- **OOM на больших моделях** → bf16 + gradient checkpointing + gradient accumulation. Если всё ещё OOM — FSDP/DeepSpeed.
- **GPU занят на 30%** → узкое место в DataLoader (`num_workers > 0`, `pin_memory=True`).
- **RNN не учится** → `clip_grad_norm_` 1.0, инициализация forget bias = 1.
- **Трансформер не сходится** → warmup, Pre-LN, gradient clipping, понизьте lr.

---

[← Назад к README курса](./README.md)
