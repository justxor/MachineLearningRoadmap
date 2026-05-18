# 01. Цифровые изображения и базовые операции

## Что такое изображение для компьютера

Картинка — это тензор: H × W × C (height × width × channels).

- **Grayscale:** H × W, значения 0-255 (uint8) или 0.0-1.0 (float).
- **RGB:** H × W × 3.
- **RGBA:** H × W × 4 (с alpha-каналом прозрачности).

**Подвох:** в PyTorch порядок осей **C × H × W**, в OpenCV/PIL **H × W × C**. Перепутали — получите кашу. Используйте `permute()` или `transpose()` явно.

## Цветовые пространства

| Пространство | Когда использовать |
|--------------|--------------------|
| **RGB** | Стандарт для DL, отображение на экране |
| **BGR** | OpenCV дефолт — постоянно конвертируйте в RGB |
| **HSV** | Сегментация по цвету (Hue stable к освещению) |
| **LAB** | Перцептуально равномерное, для color matching |
| **Grayscale** | Edge detection, OCR, ускорение |

**Конвертация:**
```python
import cv2
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
```

## Базовые операции с OpenCV и PIL

```python
import cv2
import numpy as np
from PIL import Image

# Чтение (OpenCV возвращает BGR!)
img = cv2.imread('photo.jpg')  # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Через PIL — сразу RGB
img_pil = Image.open('photo.jpg')
img_np = np.array(img_pil)  # RGB

# Resize
resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
# INTER_AREA для уменьшения, INTER_LINEAR/CUBIC для увеличения

# Crop
cropped = img[100:400, 200:500]  # [y1:y2, x1:x2]

# Flip
flipped = cv2.flip(img, 1)  # 0 — вертикально, 1 — горизонтально

# Rotate
M = cv2.getRotationMatrix2D((w/2, h/2), angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

## Форматы файлов

| Формат | Lossy/Lossless | Когда |
|--------|----------------|-------|
| **JPEG** | Lossy | Фото для web, дешёвое хранение |
| **PNG** | Lossless | Графика, прозрачность, eval-данные |
| **WebP** | Both | Web (меньше JPEG при том же качестве) |
| **TIFF** | Lossless | Медицина, профессиональная фотография |
| **BMP** | Lossless | Устаревший, не используйте |
| **HEIC** | Lossy | iPhone дефолт, проблемы с поддержкой |

**Подвох:** JPEG-артефакты от повторного сохранения накапливаются. Промежуточные результаты держите в PNG/TIFF.

## Чтение видео

```python
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # process frame (BGR)
cap.release()
```

**Альтернатива:** `decord` или `PyAV` — быстрее для batch processing видео.

## Нормализация для DL

Стандарт ImageNet (используют большинство pretrained моделей):

```python
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0, 255] → [0, 1], H×W×C → C×H×W
    normalize
])
```

**Подвох:** если ваша модель ожидает другую нормализацию (CLIP, SAM, новые модели) — используйте их статистики. Свой mean/std считайте только если pretrained веса не подходят.

## Battery: типовые операции, которые надо уметь

1. **Concatenate картинки:** `np.hstack`, `np.vstack`, `cv2.hconcat`.
2. **Конвертация цвета:** между всеми основными пространствами.
3. **Изменение яркости/контраста:** `cv2.convertScaleAbs(img, alpha=1.2, beta=10)`.
4. **Гистограмма:** `cv2.calcHist([img], [0], None, [256], [0, 256])`.
5. **Equalize histogram:** `cv2.equalizeHist` или `cv2.createCLAHE` (adaptive).
6. **Маски и bitwise операции:** `cv2.bitwise_and(img, img, mask=mask)`.
7. **Padding:** `cv2.copyMakeBorder` для аугментации до квадрата.

## Practical tips

- **Загрузка датасета медленная?** Сохраняйте в формате `webdataset` (tar) или `LMDB` — в разы быстрее, чем тысячи мелких JPEG.
- **GPU простаивает при обучении?** Bottleneck в data loading. Увеличьте `num_workers` в DataLoader, используйте `pin_memory=True`.
- **Картинки разного размера → padding или resize?** Resize дешевле, padding сохраняет aspect ratio. Для детекции — обычно padding (letterbox).
- **JPEG-чтение медленное?** Используйте `turbojpeg` или `Pillow-SIMD`.

## Антипаттерны

- Читать в `for` цикле в основном потоке → загрузка занимает 90% времени.
- Забывать BGR ↔ RGB → модель учится с одним порядком, инференс с другим.
- Использовать `cv2.resize` с дефолтным `INTER_LINEAR` для downsampling → артефакты. Используйте `INTER_AREA`.
- Хранить тысячи мелких JPEG на сетевом диске → IOPS-боттлнек.

## Задания

1. Напишите функцию, которая принимает путь к картинке и возвращает её `mean` и `std` по каждому из RGB-каналов. Проверьте на 100 случайных картинках из ImageNet — близко ли к стандартным значениям.
2. Реализуйте свою `Dataset` для PyTorch, который читает картинки из папки с метками из CSV. Поддержите кэширование в памяти для маленьких датасетов.
3. Сравните скорость чтения 1000 JPEG: через PIL, через OpenCV, через turbojpeg. Какая разница?
4. Напишите скрипт, который конвертирует папку с JPEG в WebP с заданным quality. Сравните суммарный размер.
5. Реализуйте `letterbox resize` (resize с сохранением aspect ratio + padding до квадрата). Сравните с обычным resize на тестовых картинках.
6. Прочитайте видео, выберите каждый 10-й кадр и сохраните как JPEG. Сделайте это асинхронно через `asyncio` или `concurrent.futures`.

## Чек-лист

- [ ] Понимаю разницу C×H×W и H×W×C, знаю, как конвертировать.
- [ ] Знаю, почему OpenCV возвращает BGR и где это проблема.
- [ ] Умею делать resize, crop, rotate, flip через OpenCV.
- [ ] Знаю, какую нормализацию использовать для pretrained моделей.
- [ ] Реализовал свой PyTorch Dataset с правильным data loading.

## Дальше

➡️ [02-classical-cv.md](./02-classical-cv.md) — классические алгоритмы CV: фильтры, edges, features.
