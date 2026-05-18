# 21. Безопасность и privacy в CV

CV — одна из самых privacy-sensitive областей ML. Лица, биометрия, deepfakes, surveillance. Здесь — что нужно знать инженеру в проде.

## Adversarial attacks

Малозаметные перцептивные изменения, ломающие модель.

### Типы

- **FGSM (Fast Gradient Sign Method):** добавляем ε·sign(∇loss) к изображению.
- **PGD (Projected Gradient Descent):** итеративная версия FGSM, сильнее.
- **C&W (Carlini-Wagner):** оптимизация для минимального perturbation.
- **Patch attacks:** наклейка/sticker, обманывающая детектор.
- **Black-box attacks:** без доступа к модели, через query или transfer.

### Защита

- **Adversarial training:** включаем adversarial examples в train.
- **Input preprocessing:** JPEG compression, blur — частично нейтрализует.
- **Certified defenses (randomized smoothing):** математические гарантии robustness в ограниченном радиусе.
- **Ensemble:** несколько моделей хуже атакуются одной перtubation.

**Реальность 2026:** полной защиты нет. Можно только повысить cost атаки.

```python
# Test robustness
import torchattacks
attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
adv_images = attack(images, labels)
acc_adv = (model(adv_images).argmax(1) == labels).float().mean()
```

## Face recognition: специфика

### Liveness detection

Защита от фото-атак: «это фото на телефоне, а не живой человек».

**Подходы:**
- Multi-frame: моргание, движение головы.
- 3D depth (iPhone Face ID).
- Texture analysis (фото имеет другую текстуру).
- Challenge-response: «поверни голову влево».

### Bias

Face recognition исторически имеет bias по skin tone, age, gender. Проверять обязательно:
- Validation на разнообразном датасете (FairFace, RFW).
- Per-group accuracy/FPR/FNR.
- Если деплоится в high-stakes (police, hiring) — независимый audit.

### Регулирование

- **EU AI Act:** жёсткое регулирование face recognition в общественных местах.
- **GDPR:** биометрия — особая категория.
- **State-specific (Illinois BIPA):** строгие requirements consent.

## PII detection и redaction

При обработке user-uploaded content нужно identify и protect PII:

### Faces

Detect face → blur/pixelate/replace. Стандарт для surveillance публичных.

```python
import face_recognition
boxes = face_recognition.face_locations(image)
for top, right, bottom, left in boxes:
    face = image[top:bottom, left:right]
    image[top:bottom, left:right] = cv2.GaussianBlur(face, (51, 51), 30)
```

### License plates

YOLO detector → blur. Стандарт в Street View.

### Sensitive documents

OCR → regex (passport numbers, credit cards) → redact.

### Address signs, screens, jewelry

Reducible через context-aware models или просто blanket blur backgrounds.

## Deepfake detection

Detection синтетических медиа — арм-рейс с генеративными моделями.

**Подходы:**
- **Visual artefacts:** GANs оставляют specific patterns (frequency analysis).
- **Temporal inconsistencies:** в видео — несогласованность между кадрами.
- **Biometric inconsistencies:** реальные люди blink с определёнными patterns.
- **Watermark detection:** ловим watermark (SynthID и др.).

**Tools:**
- **DeepFake-o-meter.**
- **Microsoft Video Authenticator.**

**Каверзно:** новые generative модели сильно опередили старые детекторы. Доверять одному детектору нельзя.

## Watermarking generated content

Для своих generated images:
- **Visible watermark.**
- **Invisible (SynthID Google, Stable Signature от Meta):** detectable steganography.
- **C2PA:** standard для cryptographic content provenance.

## Data privacy

### On-device processing

Когда можно — обрабатывайте на user device, не отправляйте на сервер. iOS/Android NPUs позволяют.

### Federated learning

Обучение без centralized сбора данных. Каждый device обучает локально → агрегируем gradients.

### Differential privacy

Добавление шума к gradients/features. Математические гарантии privacy.

### Анонимизация датасета

При публикации/sharing датасета:
- Blur лиц/license plates.
- Remove EXIF metadata (GPS coords!).
- K-anonymity для метаданных.

## Лицензирование данных

Где брать данные **легально**:
- **Open datasets:** COCO, ImageNet, Open Images — проверяйте лицензии (CC-BY, CC-NC, custom).
- **Synthetic data:** generated, но осторожно с GAN/diffusion training data lineage.
- **Stock photos:** Shutterstock, Adobe Stock — paid.
- **Self-collected:** with consent.

**Подвох:** scraping из интернета — серая зона. LAION-5B contains questionable content. Для commercial use — лучше custom datasets.

## CSAM (child safety)

Critical: priority 0 безопасность.

- **PhotoDNA (Microsoft):** hash-based detection known CSAM.
- **Apple NeuralHash, Google CSAI Match:** ML-based detection.
- **Zero tolerance:** немедленный block + report to NCMEC (US) или local authorities.
- **Никогда не train свою модель на CSAM** «для defense» — нелегально.

Все content platforms обязаны иметь CSAM detection.

## Регулирование

- **EU AI Act (2024+):** high-risk CV applications регулируются.
- **GDPR / CCPA:** privacy and consent.
- **Biometric laws:** state-by-state.
- **Sector-specific:** medical (HIPAA), finance (KYC).

## Антипаттерны

- **Train на scraped internet** для commercial. Юридические риски.
- **Игнорировать bias.** Особенно в high-stakes domain.
- **No liveness check** для security face recognition.
- **Storing raw images** дольше required. Сделайте retention policy.
- **Не watermarkать generative output.** Опасно для users и компании.
- **Trusting единственный deepfake детектор.** Адаптивные атаки.

## Задания

1. Запустить FGSM/PGD атаку на pretrained ResNet. Сравнить с/без adversarial training.
2. Реализовать face blurring pipeline для batch фото.
3. Реализовать PII redaction (license plates) для dashcam видео.
4. Загрузить open deepfake detector, проверить на различных source: real, FaceSwap, StableDiffusion.
5. Implement differential privacy для feature embeddings (Laplace noise).
6. Audit bias face recognition модели на FairFace dataset.

## Чек-лист

- [ ] Знаю основные adversarial attacks.
- [ ] Понимаю privacy techniques: blur, on-device, federated, DP.
- [ ] Знаю про bias и audit процессы.
- [ ] Понимаю watermarking и provenance.
- [ ] Знаю регуляции: GDPR, EU AI Act, BIPA.
- [ ] CSAM detection как priority 0.

## Дальше

Курс завершён! Переходите к капстонам:

➡️ [Капстон 1. End-to-end детектор объектов](./capstone-1-detector.md)

➡️ [Капстон 2. Real-time видео-аналитика с трекингом](./capstone-2-video-analytics.md)

➡️ [Капстон 3. Multi-modal RAG над изображениями и документами](./capstone-3-multimodal-rag.md)
