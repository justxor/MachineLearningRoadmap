# Урок 08. Attention и self-attention

> Цель урока: понять механизм внимания, который лежит в основе всех современных LLM. Реализовать scaled dot-product attention с нуля и интуитивно почувствовать, что такое Q, K, V.

## Зачем attention

В seq2seq на LSTM (перевод и подобные задачи) у декодера была одна большая проблема: вся информация об источнике сжималась в один вектор фиксированного размера. Длинные предложения теряли смысл.

**Идея attention (Bahdanau, 2014):** на каждом шаге декодер не получает один сжатый вектор, а смотрит на **все** скрытые состояния энкодера и сам **взвешенно** их комбинирует.

Через 3 года эта идея превратилась в self-attention и заменила собой RNN полностью (Vaswani et al., «Attention Is All You Need», 2017).

## Query, Key, Value — главная аналогия

Представьте поиск в Google:

- **Query** — ваш запрос («лучшие книги по ML»).
- **Key** — заголовки страниц в индексе. Поиск находит **похожие** на ваш запрос.
- **Value** — содержимое страниц. Вы получаете не заголовки, а контент.

Self-attention делает то же самое внутри последовательности. Каждый токен порождает свой запрос, ключ и значение.

## Scaled Dot-Product Attention

Формула из оригинальной статьи трансформера:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Шаг за шагом:

1. **Q · Kᵀ** — для каждого запроса считаем «похожесть» на все ключи (скалярное произведение).
2. **/√d_k** — нормировка, чтобы значения не «уезжали» при большой размерности (иначе softmax насытится и градиенты умрут).
3. **softmax** — превращаем в распределение вероятностей по позициям.
4. **× V** — взвешенная сумма value-векторов с этими весами.

## Self-attention с нуля на NumPy

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

# Игрушечный пример: 4 токена, размер эмбеддинга 8
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 8))
W_q = rng.normal(size=(8, 8))
W_k = rng.normal(size=(8, 8))
W_v = rng.normal(size=(8, 8))
out, w = self_attention(X, W_q, W_k, W_v)
print("weights shape:", w.shape)  # (4, 4) — кто на кого смотрит
```

Каждая строка матрицы `weights` — это распределение внимания одного токена по всей последовательности.

## Multi-Head Attention

Вместо одного набора `Q, K, V` — делаем **h голов**, каждая со своими маленькими `Q, K, V`. Каждая голова учится на свой аспект (одна — на синтаксис, другая — на семантику и т.д.):

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

В PyTorch:

```python
import torch, torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 64)   # batch=2, seq=10, dim=64
out, attn = mha(x, x, x)     # Q=K=V=x — это self-attention
```

## Causal mask: как сделать attention автогрессивным

В LLM декодер должен видеть только **предыдущие** токены (иначе при обучении ответ просочится в вопрос). Решение — маска: до softmax заменяем верхний треугольник матрицы `scores` на `-∞`.

```python
def causal_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ V
```

## Cross-attention

Энкодер-декодер архитектуры (T5, оригинальный трансформер для перевода) используют **cross-attention**: Q из декодера, K и V из энкодера. То есть «декодер запрашивает, на что смотреть в энкодере».

В GPT-подобных декодер-only моделях используется **только** causal self-attention.

## 8 практических заданий

1. **Self-attention руками.** Запустите код выше. Постройте heatmap матрицы `weights`. Прокомментируйте структуру.
2. **Масштабирование.** Уберите деление на `√d_k`. Постройте распределение softmax при `d_k=8, 64, 512`. Где softmax становится «one-hot»?
3. **Multi-head с нуля.** Реализуйте `MultiHeadAttention` как класс с h головами, без `nn.MultiheadAttention`. Сравните выход с PyTorch при одинаковых весах.
4. **Causal mask.** Реализуйте `causal_attention` на NumPy. Проверьте, что верхний треугольник весов = 0.
5. **Attention для классификации.** Возьмите IMDB. Соберите простую модель: embeddings → 1 слой self-attention → mean-pool → fc → 2 класса. Сравните с GRU из урока 07.
6. **Визуализация на тексте.** Обучите 1-слойный self-attention на коротких предложениях. Для конкретного предложения нарисуйте, какой токен на какой смотрит. Прокомментируйте: совпадает с синтаксисом?
7. **Эффект числа голов.** Обучите модель из задания 5 с `h = 1, 2, 4, 8, 16`. Постройте график accuracy vs h.
8. **Cross-attention.** Реализуйте простой энкодер-декодер для копирования последовательности: вход — `[1,2,3,4]`, выход тот же. Используйте causal self-attention в декодере и cross-attention на энкодер.

## Чек-лист урока

- [ ] Я могу руками вывести формулу scaled dot-product attention.
- [ ] Я понимаю, что такое Q, K, V и почему их три.
- [ ] Я знаю, зачем делить на `√d_k`.
- [ ] Я реализовал self-attention с нуля без библиотек.
- [ ] Я умею делать causal mask.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 07](./07-rnn-lstm.md) · [README курса](./README.md) · ▶︎ [Урок 09 — Transformer от начала до конца](./09-transformer.md)
