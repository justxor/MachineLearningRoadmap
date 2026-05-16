# Урок 03. Функции активации и инициализация весов

> Цель урока: понять, почему `sigmoid` уступила место `ReLU`, что такое `vanishing/exploding gradients` и как инициализация весов решает половину проблем обучения.

## Зачем вообще нелинейности

Без нелинейностей сеть из 100 слоёв эквивалентна одному линейному слою. Активация — это то, что делает нейросеть способной выражать сложные функции, и одновременно — главный источник проблем при обучении.

## Главные активации и их свойства

**Sigmoid**

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z))
$$

Историческая активация. Производная максимум `0.25` — градиенты быстро затухают при глубине. Сейчас почти не используется в скрытых слоях, остаётся только в выходе бинарного классификатора.

**Tanh**

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Центрирована в нуле — лучше, чем sigmoid, но та же проблема с насыщением.

**ReLU (Rectified Linear Unit)**

$$
\text{ReLU}(z) = \max(0, z)
$$

Стандарт де-факто. Производная: 1 если z>0, иначе 0. **Преимущества:** нет насыщения для положительных значений, считается мгновенно. **Проблема:** «мёртвые нейроны» — если веса увели нейрон в отрицательную зону, его градиент = 0 навсегда.

**Leaky ReLU / GELU / SiLU**

`Leaky ReLU`: `max(0.01·z, z)` — лечит мёртвые нейроны.
`GELU`: `z · Φ(z)` (Φ — CDF нормального распределения). Стандарт в трансформерах.
`SiLU/Swish`: `z · σ(z)`. Гладкая, часто работает лучше ReLU в современных архитектурах.

**Softmax** (для выхода многоклассового классификатора)

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Превращает любой вектор в распределение вероятностей.

## Проблема: vanishing / exploding gradients

При прямом проходе через L слоёв активации перемножаются. Если их величины систематически <1 — сигнал затухает. Если >1 — взрывается. То же самое в обратном проходе с градиентами. Это и есть **vanishing/exploding gradients**.

**Решение:** правильная инициализация + правильная активация.

## Инициализация весов

**Нулевая инициализация — нельзя.** Все нейроны слоя получат одинаковые градиенты и обучатся одинаково (проблема симметрии).

**Слишком большая случайная — нельзя.** Активации взорвутся.

**Xavier/Glorot (для tanh/sigmoid):**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

**He (для ReLU и его вариаций):**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

В PyTorch это `nn.init.xavier_normal_` и `nn.init.kaiming_normal_`.

## Демонстрация: глубокая сеть с разной инициализацией

```python
import torch, torch.nn as nn
import matplotlib.pyplot as plt

def make_net(init_fn, activation, depth=10, width=256):
    layers = []
    for i in range(depth):
        lin = nn.Linear(width, width)
        init_fn(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, activation()]
    return nn.Sequential(*layers)

x = torch.randn(64, 256)
configs = [
    ("normal(0, 1)", lambda w: nn.init.normal_(w, 0, 1), nn.ReLU),
    ("xavier", nn.init.xavier_normal_, nn.Tanh),
    ("kaiming", nn.init.kaiming_normal_, nn.ReLU),
]
for name, init, act in configs:
    net = make_net(init, act)
    with torch.no_grad():
        h = x.clone()
        stds = []
        for layer in net:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                stds.append(h.std().item())
    plt.plot(stds, label=name)
plt.yscale('log'); plt.legend(); plt.xlabel('layer'); plt.ylabel('std')
```

Вы увидите: при `normal(0,1)` std растёт экспоненциально (взрыв), при He/Xavier — остаётся стабильным. **Это и есть смысл правильной инициализации.**

## Сравнение активаций на MNIST

```python
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

# Обучите Net с torch.sigmoid, torch.tanh, F.relu, F.gelu и сравните test-accuracy.
```

На глубоких сетях разница между `sigmoid` и `ReLU` доходит до 10-20% по точности.

## 8 практических заданий

1. **График активаций.** Постройте `sigmoid, tanh, ReLU, LeakyReLU, GELU, SiLU` на `x ∈ [-5, 5]`. На отдельном графике — их производные.
2. **Vanishing на sigmoid.** Постройте сеть из 20 sigmoid-слоёв, инициализированную `N(0,1)`. Замерьте норму градиента `fc1` после одного backward. Сравните с He-инициализацией + ReLU.
3. **Воспроизведите демо.** Запустите код выше и подтвердите, что `normal(0,1)` взрывает std, а He — стабилизирует.
4. **Сравнение на MNIST.** Обучите MLP 784→256→256→10 с `sigmoid`, `tanh`, `ReLU`, `GELU`. Сделайте таблицу test-accuracy.
5. **Dead ReLU.** Намеренно инициализируйте сеть так, чтобы 80% ReLU-нейронов «умерли». Поправьте LeakyReLU. Сравните accuracy.
6. **Softmax численно стабильный.** Реализуйте `softmax(z)` так, чтобы он не переполнялся при больших `z` (подсказка: вычесть max).
7. **Своя активация.** Реализуйте `Swish(z) = z·σ(z)` как `nn.Module` с правильным обратным проходом. Обучите MLP на MNIST с ней.
8. **He vs Xavier.** Для одной и той же сети с ReLU сравните test-accuracy при инициализациях Xavier и He. На какой эпохе становится видна разница?

## Чек-лист урока

- [ ] Я могу нарисовать график 6 главных активаций и их производных.
- [ ] Я понимаю, что такое vanishing/exploding gradients геометрически.
- [ ] Я знаю, какую инициализацию использовать для ReLU и для Tanh.
- [ ] Я воспроизвёл демо с std по слоям и видел разницу.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 02](./02-mlp-backprop.md) · [README курса](./README.md) · ▶︎ [Урок 04 — Регуляризация](./04-regularization.md)
