# Урок 07. RNN, LSTM, GRU и работа с последовательностями

> Цель урока: понять, почему MLP/CNN не подходят для последовательностей переменной длины, как работает рекуррентный нейрон и зачем понадобились LSTM/GRU. И почему всё это в итоге проиграло трансформерам — но осталось важной частью фундамента.

## Зачем нужны рекуррентные сети

Текст, аудио, цены акций, ДНК — данные, где **порядок** имеет значение и длина может быть произвольной. MLP требует фиксированного размера входа. CNN видит локальный контекст, но плохо моделирует длинные зависимости. RNN решает обе проблемы.

## Рекуррентный нейрон

Главная идея: на каждом шаге мы обновляем **скрытое состояние** `h_t`, используя текущий вход `x_t` и предыдущее состояние `h_{t-1}`:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

Те же веса используются на каждом шаге → параметров мало, а длина последовательности может быть любой.

## RNN с нуля на NumPy

```python
import numpy as np

def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

def rnn_forward(X, W_xh, W_hh, b_h):
    T = len(X)
    h = np.zeros(W_hh.shape[0])
    H = []
    for t in range(T):
        h = rnn_step(X[t], h, W_xh, W_hh, b_h)
        H.append(h)
    return np.array(H)
```

## Проблема: vanishing gradients через время

Backpropagation Through Time (BPTT) разворачивает RNN на T шагов и считает градиенты. На каждом шаге градиент умножается на `W_hh`. Если её спектральный радиус < 1 — градиенты затухают экспоненциально, > 1 — взрываются. На практике RNN тяжело учится зависимостям длиннее ~20 шагов.

**Решения:**

- **Gradient clipping** — обрезать норму градиента: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
- **LSTM/GRU** — архитектуры с явным механизмом сохранения состояния.

## LSTM

Long Short-Term Memory (Hochreiter & Schmidhuber, 1997). Главная идея — добавить **ячейку памяти** `c_t`, поток которой регулируется тремя обучаемыми «воротами»:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}
$$

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}
$$

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate)}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}
$$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

**Интуиция:** forget gate решает, что забыть из памяти; input gate — что добавить; output gate — что отдать на выход. Поток `c_t` идёт почти линейно через время → градиенты не затухают.

## GRU

Gated Recurrent Unit — упрощённый LSTM с двумя воротами вместо трёх:

$$
r_t = \sigma(W_r [h_{t-1}, x_t]), \quad z_t = \sigma(W_z [h_{t-1}, x_t])
$$

$$
\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t])
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

Параметров меньше, обычно сходится быстрее. На большинстве задач GRU ≈ LSTM по качеству. Если сомневаетесь — начните с GRU.

## RNN/LSTM/GRU в PyTorch

```python
import torch, torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.LSTM(hidden, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        x = self.embed(x)
        out, state = self.rnn(x, state)
        return self.fc(out), state

# Обучение: предсказываем следующий символ
# Loss: cross_entropy между out и сдвинутым на 1 x
```

## Применения RNN (исторические и актуальные)

- **Языковые модели до 2017** — почти все на LSTM.
- **Машинный перевод seq2seq** — encoder LSTM + decoder LSTM + attention (Bahdanau, 2014). Прямой предшественник трансформера.
- **Speech recognition** — DeepSpeech (Baidu), LAS (Google).
- **Time series** — прогнозирование, аномалии. Здесь RNN до сих пор конкурентоспособны.
- **Online inference** — модели, которые **видят поток в реальном времени** (биржа, телеметрия). RNN работают по одному шагу, трансформер пересчитывает контекст.

## Почему трансформеры победили

- **Параллелизация:** RNN считается последовательно (T шагов), трансформер — всё параллельно за один матричный умножения.
- **Длинные зависимости:** attention видит любую пару позиций напрямую.
- **Масштабирование:** в RNN при увеличении hidden state вычисления растут линейно, а память — квадратично; в трансформере — наоборот, и это хорошо легло на GPU.

Но это не значит «забудьте RNN». Понимание seq2seq + attention — прямой путь к интуитивному пониманию трансформера (урок 09).

## 8 практических заданий

1. **RNN forward с нуля.** Реализуйте `rnn_forward` на NumPy. Прогоните на синусоиде 100 точек.
2. **Свой BPTT.** Реализуйте обратное распространение через время для RNN на одном примере. Сравните градиенты с PyTorch autograd.
3. **Char-RNN на Шекспире.** Обучите `CharRNN` на тексте `tinyshakespeare`. Сгенерируйте 500 символов с температурой 0.8.
4. **LSTM vs RNN.** На той же задаче сравните `nn.RNN`, `nn.LSTM`, `nn.GRU` по loss и качеству генерации.
5. **Gradient clipping.** Уберите `clip_grad_norm_`. Покажите, что обучение RNN расходится. Верните клиппинг — стабильно.
6. **Sentiment classification.** Возьмите IMDB. Обучите GRU-классификатор. Достигните accuracy > 85%.
7. **Двунаправленный.** Замените `nn.LSTM(...)` на `nn.LSTM(..., bidirectional=True)`. Сравните метрики.
8. **Time series.** Обучите GRU на прогнозирование цены акции (например, дневная цена за последний год). Сравните с baseline «предсказать последнее значение».

## Чек-лист урока

- [ ] Я могу написать RNN-step с нуля.
- [ ] Я понимаю vanishing gradients через время и зачем нужен gradient clipping.
- [ ] Я могу нарисовать схему LSTM с тремя воротами.
- [ ] Я обучил char-RNN и она генерирует осмысленный текст.
- [ ] Я понимаю, почему трансформер заменил RNN в NLP.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 06](./06-cnn.md) · [README курса](./README.md) · ▶︎ [Урок 08 — Attention](./08-attention.md)
