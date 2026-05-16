# Урок 02. MLP и обратное распространение

> Цель урока: собрать многослойный перцептрон руками, разобрать backpropagation как цепное правило и обучить XOR — задачу, которую один нейрон решить не может.

## Что такое MLP

**Multi-Layer Perceptron** — это последовательность линейных слоёв с нелинейностями между ними:

$$
\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \hat{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3
$$

Без нелинейности σ суперпозиция любых линейных слоёв — снова линейная модель. **Именно нелинейность делает нейросеть нейросетью.**

**Универсальная теорема аппроксимации**: MLP с одним скрытым слоём и достаточным числом нейронов может приблизить любую непрерывную функцию на компактном множестве. Это не значит «один слой решает всё» — на практике глубокие сети учатся быстрее и обобщают лучше.

## Backpropagation — это просто цепное правило

Допустим, у нас сеть `x → z = Wx + b → h = σ(z) → L(h, y)`. Мы хотим найти `∂L/∂W`. По цепному правилу:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

Backprop = вычислить эти производные **в обратном порядке** через сеть, переиспользуя промежуточные значения. Никакой магии — школьная математика, аккуратно применённая.

## MLP с нуля для XOR (NumPy)

```python
import numpy as np
rng = np.random.default_rng(0)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(z): return 1/(1+np.exp(-z))
def dsigmoid(a): return a*(1-a)  # производная через готовое значение sigmoid

# 2 → 4 → 1
W1 = rng.normal(0, 1, (2, 4))
b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1, (4, 1))
b2 = np.zeros((1, 1))

lr = 0.5

for epoch in range(5000):
    # forward
    z1 = X @ W1 + b1
    h1 = sigmoid(z1)
    z2 = h1 @ W2 + b2
    y_hat = sigmoid(z2)

    # loss
    loss = -np.mean(y*np.log(y_hat+1e-9) + (1-y)*np.log(1-y_hat+1e-9))

    # backward
    dz2 = (y_hat - y) / len(X)        # ∂L/∂z2 для BCE+sigmoid
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    dh1 = dz2 @ W2.T
    dz1 = dh1 * dsigmoid(h1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

print("loss:", loss)
print("predictions:", y_hat.round(2).ravel())
```

Если всё правильно — после ~2000 эпох вы увидите что-то вроде `[0.02, 0.98, 0.98, 0.02]`. XOR побеждён.

## То же в PyTorch (но руками понимая, что внутри)

```python
import torch, torch.nn as nn

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x))))

model = MLP()
opt = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.BCELoss()

for epoch in range(5000):
    y_hat = model(X_t)
    loss = loss_fn(y_hat, y_t)
    opt.zero_grad(); loss.backward(); opt.step()
```

`loss.backward()` делает в точности то же, что наш NumPy-код. Просто вы это уже видели.

## Численная проверка градиентов

Перед тем, как обучать что-то серьёзное, **всегда** проверяйте свои градиенты численно:

```python
def numerical_grad(f, W, eps=1e-5):
    g = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = W[idx]
        W[idx] = old + eps; lp = f()
        W[idx] = old - eps; lm = f()
        W[idx] = old
        g[idx] = (lp - lm) / (2*eps)
        it.iternext()
    return g
```

Если ваш аналитический градиент отличается от численного на >1e-5 — где-то ошибка в backward.

## 8 практических заданий

1. **XOR.** Запустите код выше. Постройте график loss от эпохи. Объясните, почему loss падает «ступеньками».
2. **Глубже.** Сделайте сеть 2→4→4→1. Сравните скорость сходимости с 2→4→1.
3. **Свой backprop.** Реализуйте сеть 2→3→2 (классификация 2 классов через softmax) без autograd. Используйте кросс-энтропию.
4. **Сравнение с PyTorch.** Сравните loss и accuracy ручной реализации и PyTorch на 1000 эпохах. Должны совпадать с точностью 0.001.
5. **Проверка градиентов.** Реализуйте `numerical_grad` и проверьте свои `dW1, dW2` на одной итерации.
6. **MNIST.** Возьмите MNIST (можно через `torchvision`), сделайте MLP 784→128→64→10. Достигните accuracy > 0.97 на тесте.
7. **Влияние ширины слоя.** Для MNIST постройте график test-accuracy в зависимости от ширины скрытого слоя: 16, 64, 256, 1024. Где плато?
8. **Vanishing gradient.** Постройте сеть из 10 `sigmoid`-слоёв. Покажите, что градиенты в первых слоях стремятся к 0. Это мотивация урока 03.

## Чек-лист урока

- [ ] Я могу руками вывести backprop для сети из 2 слоёв.
- [ ] Я понимаю, зачем нужна нелинейность.
- [ ] Я реализовал MLP без autograd и сравнил с PyTorch.
- [ ] Я обучил MLP на MNIST с accuracy > 0.97.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 01](./01-perceptron.md) · [README курса](./README.md) · ▶︎ [Урок 03 — Функции активации и инициализация](./03-activations-init.md)
