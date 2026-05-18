# 12. Vision Transformers (ViT)

С 2020 года ViT — серьёзная альтернатива CNN. С 2022 — стандарт для multi-modal, foundation models, любых задач с большими данными.

## ViT: как работает

1. Картинка → разбиваем на патчи (16×16).
2. Каждый патч → linear projection → embedding.
3. + positional encoding.
4. + CLS token (классификационный токен).
5. Standard transformer encoder.
6. CLS token → классификация.

```
Картинка 224×224 → 196 патчей (14×14 grid) → последовательность 197 токенов (196 + CLS)
                → transformer (depth 12, heads 12) → CLS output → classifier head
```

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

## Сила ViT

- **Глобальный context.** Каждый патч сразу видит все остальные. Long-range dependencies проще.
- **Хорошее масштабирование.** При увеличении данных и параметров ViT масштабируется лучше CNN.
- **Universality.** ViT-архитектура единая для CV, NLP, мультимодальности.

## Слабости ViT

- **Прожорлив до данных.** Без 100M+ patches in pretrain хуже CNN.
- **Менее efficient на high-res.** O(n²) от числа патчей.
- **Слабее inductive bias.** CNN «знает» про локальность встроенно, ViT нет.

## Семейство ViT

| Модель | Год | Особенность |
|--------|-----|-------------|
| **ViT** (Google) | 2020 | Оригинал |
| **DeiT** (Meta) | 2020 | Distillation + augmentations, не нужен JFT |
| **Swin Transformer** | 2021 | Shifted windows, hierarchical, для детекции/сегментации |
| **MAE** (Meta) | 2021 | Masked autoencoder pretrain, self-supervised |
| **DINOv2** | 2023 | Self-supervised, лучшие features для downstream |
| **EVA** | 2022 | Огромные ViT (1B+) pretrain на CLIP+MAE |

## Swin Transformer

Хитрость: вместо global attention на всех патчах — local windows + shifting between layers. O(n) вместо O(n²).

```python
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
```

Стандарт для object detection и segmentation на ViT-подложке. Используется в Mask2Former и многих SOTA.

## DINOv2: лучший self-supervised на сегодня

Обучен на 142M картинок без меток. Features универсальны:
- Linear probe на ImageNet ≈ supervised baselines.
- Отличный для downstream (детекция, depth, segmentation) без fine-tune backbone.
- Embedding-based retrieval работает «из коробки».

```python
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
features = dinov2(images)  # global CLS embedding
patch_features = dinov2.forward_features(images)['x_norm_patchtokens']  # per-patch
```

**Применения:**
- Image retrieval / similarity search.
- Linear probing на новой задаче.
- Few-shot classification (k-NN на features).
- Depth, semantic segmentation как auxiliary tasks без fine-tuning backbone.

## MAE: pre-training рецепт

Маскируем 75% патчей картинки, обучаем восстанавливать. Encoder видит только видимые патчи (дешево), decoder восстанавливает.

После pretrain — decoder выбрасывается, encoder используется как backbone.

**Качество:** один из самых эффективных способов pretrain ViT при ограниченных данных (миллионы, а не миллиарды).

## ViT vs CNN: choosing

| Сценарий | Выбор |
|----------|-------|
| Малый датасет (< 5K), нет pretrain | **CNN (ConvNeXt, EfficientNet)** |
| Средний датасет с pretrain | **ConvNeXt или ViT-Base** |
| Multi-modal, foundation models | **ViT-based (CLIP, BLIP, LLaVA)** |
| Detection / segmentation | **Swin или ViT-based Mask2Former** |
| Best features без обучения | **DINOv2** |
| Edge / mobile | **CNN (MobileNet) или MobileViT** |

## ViT для дешёвых задач

Маленькие ViT (ViT-Tiny, ViT-Small) могут работать на mobile. **MobileViT** комбинирует CNN и attention для edge.

## Patch size choice

- **16×16:** стандарт. 196 патчей для 224×224.
- **14×14:** DINOv2, чуть лучше качество, чуть медленнее.
- **8×8:** для high-res задач (detection), но 4x больше patches → 16x дороже attention.

## Position encoding

- **Learned positions:** простой, но не экстраполируется на разрешение, отличное от train.
- **2D sin/cos:** математический, экстраполируется.
- **RoPE 2D:** современный, лучше всего работает на variable resolution.

## Antipatterns

- **Train ViT с нуля на 10K examples.** Не хватит данных, переобучится. CNN сильнее.
- **Игнорировать DINOv2** для retrieval/embedding задач — он SOTA practically out-of-the-box.
- **Свой attention с нуля** в production. Используйте `F.scaled_dot_product_attention` (включает FlashAttention автоматом).

## Задания

1. Fine-tune ViT-Base на своём датасете, сравнить с ConvNeXt-Small по accuracy и latency.
2. Использовать DINOv2 features для image retrieval: получить embeddings 1000 картинок, найти похожие на новое query.
3. Реализовать MAE pre-training на маленьком custom датасете. Сравнить downstream fine-tune с/без MAE pretrain.
4. Сравнить Swin Transformer и Mask2Former для детекции на COCO subset.
5. Визуализировать attention maps ViT на тестовой картинке. Какие части модель «смотрит»?
6. Использовать DINOv2 для linear probing на 100 примерах. Сравнить с fine-tune ResNet50.

## Чек-лист

- [ ] Понимаю архитектуру ViT и patch tokenization.
- [ ] Знаю Swin, DINOv2, MAE.
- [ ] Умею использовать DINOv2 features для retrieval/probing.
- [ ] Понимаю, когда ViT > CNN и наоборот.

## Дальше

➡️ [13-clip-multimodal.md](./13-clip-multimodal.md) — CLIP и multi-modal модели.
