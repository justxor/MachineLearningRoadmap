# Урок 09. Transformer от начала до конца

> Цель урока: собрать минимальный, но **полный** GPT-подобный трансформер на PyTorch с нуля и обучить его на тексте Шекспира. После этого урока вы перестанете воспринимать LLM как «магию».

## Что такое трансформер

Архитектура из статьи «Attention Is All You Need» (2017). Полностью заменила RNN в NLP, потом — CNN во многих задачах CV (ViT), теперь — основа всех LLM (GPT, LLaMA, Claude, Gemini).

**Главные составляющие:**

1. **Token embeddings** + **positional embeddings** — превращаем токены в векторы.
2. **N идентичных блоков**, каждый:
   - LayerNorm → Multi-head self-attention → residual.
   - LayerNorm → Feed-forward (MLP) → residual.
3. **Финальный LayerNorm** + линейный проекционный слой → логиты по словарю.

Современная норма (Pre-LN): LayerNorm применяется **до** attention/FFN, а не после. Это стабилизирует обучение глубоких трансформеров.

## Positional embeddings: зачем

Self-attention сама по себе **permutation-invariant** — если перемешать токены, attention выдаст то же самое (с точностью до перестановки). А для языка порядок критичен. Решение — добавить к токен-эмбеддингу позиционный эмбеддинг.

Варианты:

- **Sinusoidal** (оригинал) — несколько синусов разных частот.
- **Learned** (GPT) — обычные эмбеддинги, инициализируются случайно и обучаются.
- **Rotary (RoPE)** — встраиваются прямо в Q и K, инвариантны к сдвигу. Стандарт в LLaMA, Mistral, и т.д.

## Полный код мини-GPT (~100 строк)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, T, dh)
        att = (q @ k.transpose(-2, -1)) / self.d_head ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=1.0, top_k=None):
        for _ in range(max_new):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
```

## Обучение на Шекспире

```python
# Скачайте tiny shakespeare:
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt') as f:
    text = f.read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(block=128, batch=32):
    ix = torch.randint(0, len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i+block] for i in ix])
    y = torch.stack([data[i+1:i+block+1] for i in ix])
    return x, y

model = MiniGPT(vocab=len(chars), max_len=128)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(5000):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(step, loss.item())

# генерация
start = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(start, max_new=300, temperature=0.8, top_k=50)
print(''.join(itos[i.item()] for i in out[0]))
```

После 5000 шагов на CPU вы получите узнаваемый псевдо-шекспировский текст. На бесплатном GPU в Colab — за 10-15 минут.

## Что внутри FFN

Внутри блока, помимо attention, — простой 2-слойный MLP, расширяющий размерность в 4 раза (отсюда `mlp_ratio=4`). По параметрам это **больше**, чем attention. Многие исследования показывают, что FFN-слои хранят **факты**, а attention — **отношения**. Изменение FFN — основной механизм fine-tuning знаний модели.

## Масштабирование: что меняется на больших моделях

- **Sparse attention** — Flash Attention, BlockSparse: экономят память на длинных последовательностях.
- **MoE (Mixture of Experts)** — несколько FFN, на каждый токен активируется только часть. Так работают Mixtral, DeepSeek-V3 и др.
- **RoPE / ALiBi** — позиционные кодировки, лучше переносящиеся на длинные контексты.
- **Grouped-Query Attention (GQA)** — меньше K/V голов, чем Q. Экономит память KV-кеша.

Понимание мини-GPT — путь к пониманию всего этого. Архитектурно LLaMA-3 70B и MiniGPT отличаются количеством параметров и парой оптимизаций, но не идеей.

## 8 практических заданий

1. **Запуск MiniGPT.** Запустите код. Достигните loss < 1.5 на Шекспире. Сгенерируйте 300 символов.
2. **Анализ внимания.** На обученной модели вытащите матрицу attention из последнего блока на конкретном предложении. Нарисуйте heatmap.
3. **Эффект глубины.** Сравните MiniGPT с `n_layers = 1, 2, 4, 8` при одинаковом числе параметров. Постройте кривые loss.
4. **Sinusoidal vs learned positional.** Замените learned positional embeddings на sinusoidal. Сравните val-loss.
5. **Top-k / top-p sampling.** Реализуйте nucleus sampling (`top_p`) и сравните с top-k. Какой даёт более интересный текст?
6. **Свой токенизатор.** Замените char-level токенизацию на BPE (`tokenizers` от HuggingFace). Перетренируйте. Сравните качество.
7. **Inference speed.** Замерьте время генерации 100 токенов без KV-кеша и с KV-кешем (можно через `use_cache=True` или вручную). Объясните разницу.
8. **Свой fine-tune.** Возьмите GPT-2 small из `transformers`, дообучите на корпусе своих текстов (например, своих заметок). Сгенерируйте.

## Чек-лист урока

- [ ] Я могу нарисовать архитектуру одного блока трансформера по памяти.
- [ ] Я понимаю, зачем positional embeddings и где они добавляются.
- [ ] Я обучил MiniGPT с нуля и получил осмысленный текст.
- [ ] Я понимаю разницу между encoder-only, decoder-only и encoder-decoder.
- [ ] Я знаю, чем GPT отличается от BERT архитектурно.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 08](./08-attention.md) · [README курса](./README.md) · ▶︎ [Урок 10 — Embeddings и токенизация](./10-embeddings-tokenization.md)
