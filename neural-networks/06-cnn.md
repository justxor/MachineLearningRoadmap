# Урок 06. CNN с нуля: свёртки, пулинг, классификация изображений

> Цель урока: понять, что такое свёртка и пулинг как операции над изображением, собрать первую CNN руками и обучить её на CIFAR-10 до accuracy > 80%.

## Зачем CNN, если есть MLP

Картинка 256×256×3 = 196 608 признаков. Полносвязный слой на такой вход — миллионы параметров **только в первом слое**. Это:

- Дорого по памяти и вычислениям.
- Игнорирует пространственную структуру (соседний пиксель похож на текущий).
- Не инвариантно к сдвигу (если кошка сдвинется на 5 пикселей — модель её не узнает).

**Свёртка решает все три проблемы.**

## Свёртка: операция, формула, интуиция

Сворачиваем входное изображение с **ядром** `K` (обучаемая матрица обычно 3×3 или 5×5):

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)
$$

Каждый выходной пиксель = взвешенная сумма входных пикселей в окрестности. Параметров — только размер ядра (например, 3×3×3 = 27 для RGB), независимо от размера картинки.

**Что учат свёртки:** в первых слоях — края, углы, цветовые градиенты. Дальше — текстуры. Глубже — части объектов. На выходе — целые объекты. Это видно, если визуализировать активации (см. задание 7).

## Ключевые параметры свёртки

- **kernel_size** — размер ядра. Стандарт — 3.
- **stride** — шаг. `stride=2` уменьшает разрешение в 2 раза.
- **padding** — добавление нулей по краям, чтобы сохранить размер.
- **in_channels / out_channels** — сколько каналов на входе и выходе. Каждый выходной канал = свой обучаемый фильтр.

**Формула выходного размера:**

$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \cdot \text{pad} - \text{kernel}}{\text{stride}} \right\rfloor + 1
$$

## Свёртка с нуля на NumPy

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    H, W = image.shape
    kH, kW = kernel.shape
    if padding > 0:
        image = np.pad(image, padding)
        H, W = image.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[i, j] = (patch * kernel).sum()
    return out

# Детектор вертикальных краёв (фильтр Собеля)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
```

Запустите это на чёрно-белой картинке — вы своими глазами увидите, как ядро «вытягивает» вертикальные края.

## Пулинг

**Max pooling** — берём максимум в окне 2×2:

$$
y_{i,j} = \max(x_{2i, 2j}, x_{2i+1, 2j}, x_{2i, 2j+1}, x_{2i+1, 2j+1})
$$

Уменьшает разрешение в 2 раза, делает сеть инвариантнее к мелким сдвигам, экономит вычисления. В современных архитектурах часто заменяется на свёртку со `stride=2`.

## Первая CNN на PyTorch (CIFAR-10)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 16x16x32 -> 16x16x64 (после pool1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 -> 4
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
```

С аугментацией (`RandomHorizontalFlip`, `RandomCrop(32, padding=4)`) + AdamW + cosine schedule эта сеть берёт ~85% на CIFAR-10 за 30-50 эпох.

## Знаковые архитектуры — что важно знать

- **LeNet (1998)** — первая CNN для распознавания цифр.
- **AlexNet (2012)** — победила ImageNet, запустила deep learning.
- **VGG (2014)** — показала, что глубже = лучше (до определённого предела).
- **ResNet (2015)** — **skip connections** `y = F(x) + x` — теперь можно обучать сети глубиной 100+. Главное архитектурное открытие десятилетия. Все современные CNN используют residual блоки.
- **EfficientNet (2019)** — compound scaling.
- **ConvNeXt (2022)** — современный ответ ViT, доказал, что CNN ещё рано хоронить.

## 8 практических заданий

1. **Свёртка руками.** Реализуйте `conv2d` на NumPy. Примените 3 разных ядра к одной картинке: blur, edge-x, edge-y. Сохраните результаты.
2. **Output shape.** Дано: `Conv2d(3, 16, kernel=5, stride=2, padding=2)`. Входное изображение `3×64×64`. Какой shape выхода? Проверьте кодом.
3. **Receptive field.** Для сети из 3 `Conv 3×3` + `MaxPool 2×2 + Conv 3×3` посчитайте, какую область входа «видит» один пиксель выхода.
4. **CNN на CIFAR-10.** Запустите `SimpleCNN`. Достигните 80% test-accuracy. Постройте train/val графики.
5. **Residual blocks.** Добавьте к `SimpleCNN` skip connections. Сравните сходимость.
6. **Сравнение с MLP.** Замените свёртки на полносвязные слои с тем же числом параметров. Сравните accuracy и время обучения на CIFAR-10.
7. **Визуализация фильтров.** После обучения нарисуйте веса первого слоя `conv1` как 32 картинки 3×3. Должны быть видны ориентированные края.
8. **Transfer learning.** Возьмите `torchvision.models.resnet18(pretrained=True)`, замените последний слой на 10 классов CIFAR-10, дообучите. Сравните accuracy с `SimpleCNN` за то же время.

## Чек-лист урока

- [ ] Я могу руками посчитать output shape любого `Conv2d`.
- [ ] Я понимаю, чем свёртка экономит параметры по сравнению с MLP.
- [ ] Я реализовал свёртку без библиотек.
- [ ] Я обучил CNN на CIFAR-10 с accuracy > 80%.
- [ ] Я понимаю, что такое residual connections и зачем они нужны.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 05](./05-optimizers.md) · [README курса](./README.md) · ▶︎ [Урок 07 — RNN, LSTM, GRU](./07-rnn-lstm.md)
