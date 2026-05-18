# 02. Классические алгоритмы CV

Классика не умерла. Знание классики:
- Помогает понять, **что делает CNN изнутри** (свёртка = классический фильтр).
- Незаменима для preprocessing, debug, lightweight задач без GPU.
- Часть многих pipeline и в 2026 (OpenCV в каждом prod-сервисе).

## Свёртки и фильтры

Свёртка — основа всего: ядро (kernel) скользит по картинке, считая взвешенную сумму пикселей.

**Главные фильтры:**

```python
import cv2
import numpy as np

# Размытие
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
median = cv2.medianBlur(img, 5)  # для соль-перец шума
bilateral = cv2.bilateralFilter(img, 9, 75, 75)  # сохраняет границы

# Резкость (unsharp masking)
gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

# Sobel — градиенты
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
```

**Связь с DL:** свёрточные фильтры в первых слоях CNN после обучения часто выглядят как Gabor-фильтры — то, что инженеры CV проектировали руками 30 лет.

## Edge detection

**Canny — золотой стандарт:**

```python
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
```

**Алгоритм Canny:**
1. Сглаживание Gaussian.
2. Sobel для градиентов.
3. Non-maximum suppression (тонкие линии).
4. Двойной порог + hysteresis.

**Подвох:** thresholds подбирать экспериментально. Auto: `v = np.median(gray); cv2.Canny(gray, 0.66*v, 1.33*v)`.

## Morphological operations

Работают с бинарными масками (после thresholding или сегментации):

```python
kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(binary_mask, kernel, iterations=1)  # «разрастание»
erosion = cv2.erode(binary_mask, kernel, iterations=1)    # «съёживание»

opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # erosion → dilation, убирает мелкий шум
closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # dilation → erosion, заполняет дыры
```

**Когда используется:** postprocessing масок сегментации, OCR (соединение букв), удаление мелкого шума.

## Thresholding и сегментация по цвету

```python
# Простой порог
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu — автоматический порог
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive — локальный порог (для неравномерного освещения)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# Сегментация по HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower=(35, 50, 50), upper=(85, 255, 255))  # зелёный
```

## Feature detection: SIFT, ORB

**Когда нужно:** matching между картинками, panoramic stitching, AR, без обучения нейросети.

```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

**SIFT vs ORB:**
- **SIFT:** точнее, инвариантен к scale/rotation, лицензия теперь свободная.
- **ORB:** быстрее, бинарные дескрипторы, real-time.

**В 2026:** для большинства задач **superglue / LightGlue / DISK** (нейросетевые) бьют классику. Но SIFT/ORB ещё актуальны для embedded и быстрых задач.

## Homography и панорамы

Если знаете 4+ соответствующих точек между двумя изображениями — можете найти преобразование (homography):

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Применение
warped = cv2.warpPerspective(img1, H, (w, h))
```

**RANSAC** — критично: убирает outlier matches.

**Применения:** stitching, AR-overlays, document scanning (выпрямление перспективы).

## Contour detection

```python
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)
    x, y, w, h = cv2.boundingRect(c)
    # Approximation для polygons (e.g., document corners)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, closed=True)
```

## Template matching

```python
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.8)
```

**Когда работает:** ровно тот же объект, тот же размер, та же ориентация. Иначе — никак.

## Hough Transform

Поиск линий и окружностей:

```python
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                        minLineLength=50, maxLineGap=10)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                            param1=100, param2=30, minRadius=10, maxRadius=100)
```

**Применения:** обнаружение дорожной разметки, штрих-кодов, монет, документов (rectangle detection).

## Когда классика vs DL

| Задача | Классика подходит | Лучше DL |
|--------|-------------------|----------|
| Preprocessing | ✅ | — |
| Document scanning (выпрямление) | ✅ | — |
| QR/штрих-коды | ✅ (pyzbar) | — |
| Color-based сегментация (известный цвет) | ✅ | — |
| OCR простых сценариев | классика умерла | ✅ |
| Object detection в complex scenes | ❌ | ✅ |
| Face recognition | ❌ | ✅ |
| Anything in the wild | ❌ | ✅ |

**Правило:** есть **structured environment** (фикс. освещение, угол, фон) → начните с классики. Иначе сразу DL.

## Антипаттерны

- Hand-tuning thresholds для production — работает один день, ломается на следующий.
- SIFT для real-time на CPU без оптимизации — медленно.
- Игнорировать классику и сразу нейросеть на простую задачу — overkill.
- Confound MORPH_OPEN и MORPH_CLOSE.

## Задания

1. Реализуйте **document scanner**: на вход фото листа A4 под углом → выровнять перспективу (поиск 4 углов через contours + cv2.warpPerspective).
2. Напишите **детектор номера машины** через классику: серый → blur → Canny → contours → фильтр прямоугольников нужного aspect ratio. Сравните с YOLO по точности.
3. Реализуйте **stitching двух картинок в панораму** через SIFT + RANSAC + warpPerspective.
4. Сравните **edge detection** на одной и той же картинке: Sobel, Laplacian, Canny. Какой результат для разных типов сцен?
5. Реализуйте **подсчёт монет на столе** через HoughCircles + морфологию.
6. Возьмите фото текста и постройте pipeline preprocessing для OCR: greyscale → denoise → adaptive threshold → морфология. Прогон через Tesseract до/после.

## Чек-лист

- [ ] Понимаю, что такое свёртка и как её применять руками.
- [ ] Умею делать Canny edges с правильными порогами.
- [ ] Знаю морфологические операции и когда их применять.
- [ ] Понимаю, что такое homography, могу выровнять документ.
- [ ] Умею matching через SIFT/ORB + RANSAC.

## Дальше

➡️ [03-augmentations.md](./03-augmentations.md) — аугментации и data pipelines для CV.
