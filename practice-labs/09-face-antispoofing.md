# 👤 Лаба 09: Face anti-spoofing с метриками FAR/FRR 🔴

## Цель

Обучить модель отличать живое лицо от фото/видео-спуфа. Научиться работать с FAR/FRR/EER, ROC при сильном дисбалансе и cross-dataset generalization.

## Датасет

- CelebA-Spoof, OULU-NPU, SiW (по запросу).
- Открытые: CASIA-FASD subset.
- Свой: снять 100 видео (живых и с телефона по картинке).

## Минимальный пайплайн

1. Face detection (MediaPipe / MTCNN) → crop.
2. Train/val сплит по субъектам.
3. Базовая модель: ResNet-18 binary.
4. Аугментации: ColorJitter, RandomCrop, GaussianBlur (имитация печатных фото).
5. Оценка FAR/FRR при разных порогах.
6. ROC + нахождение EER.

## Код: FAR/FRR

```python
def far_frr(y_true, y_score, threshold):
    pred = (y_score >= threshold).astype(int)
    # 1 = spoof, 0 = live
    far = ((pred == 1) & (y_true == 0)).sum() / max(1,(y_true == 0).sum())
    frr = ((pred == 0) & (y_true == 1)).sum() / max(1,(y_true == 1).sum())
    return far, frr
```

## Метрики

- FAR (False Accept Rate) — пропустили спуф.
- FRR (False Reject Rate) — отвергли живого.
- EER — точка FAR=FRR.
- HTER = (FAR+FRR)/2 при фиксированном пороге.
- Cross-dataset: train OULU, test CASIA.

## Расширения

- Depth-map auxiliary task.
- Temporal model (rPPG, optical flow).
- Domain adaptation (DANN).
- Active learning на сложных случаях.
- Adversarial спуфы: проверить устойчивость.

## Критерии приёмки

- [ ] ROC + EER в отчёте.
- [ ] FAR@FRR=1% в таблице.
- [ ] Cross-dataset результат (хотя бы 2 датасета).
- [ ] EER <5% на in-domain valid.
- [ ] Разбор сложных кейсов в отчёте.

## Анти-паттерны

- ❌ Оценка по accuracy.
- ❌ Сплит по картинкам, не по субъектам.
- ❌ Тест только на одном датасете (нет generalization).
- ❌ Игнор сложных спуфов: maska 3D, video replay.

---

[← Назад к Practice Labs](./README.md)
