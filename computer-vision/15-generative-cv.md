# 15. Генеративные модели для CV

Не «как нарисовать арт», а как использовать generative модели для прикладных задач: super-resolution, inpainting, стилизация, синтетические данные.

## Stable Diffusion и Flux: основы

**Stable Diffusion** — latent diffusion model:
1. VAE encoder сжимает 512×512 → 64×64 latent.
2. U-Net (или DiT) с text conditioning делает denoising.
3. VAE decoder восстанавливает картинку.

**Flux** (Black Forest Labs, 2024) — современный SOTA, использует DiT и Flow Matching. Лучше качество, проще архитектура.

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe("a cat wearing sunglasses", num_inference_steps=50, guidance_scale=3.5).images[0]
```

## Прикладные задачи

### Super-resolution

Увеличить разрешение картинки без потери качества.

- **Real-ESRGAN:** GAN-based, быстро, проверено.
- **SwinIR:** transformer-based, лучше качество.
- **Latent SR:** через diffusion, для максимального качества.

```python
# Real-ESRGAN
from realesrgan import RealESRGANer
upsampler = RealESRGANer(scale=4, model_path='RealESRGAN_x4plus.pth', ...)
output, _ = upsampler.enhance(image, outscale=4)
```

### Inpainting

Восстановить выделенную область («магическая кисть» в Photoshop).

- **Stable Diffusion Inpainting:** generic.
- **LaMa:** специализирован, быстрее.

```python
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
result = pipe(prompt="a flower", image=image, mask_image=mask).images[0]
```

### Outpainting

Расширение картинки за её границы. Используется ту же inpainting, но с маской «снаружи» оригинала.

### Background removal

- **rembg** (открытый, U-Net based).
- **BiRefNet, BRIA RMBG-1.4** — современные SOTA для фотобэкграундов.

### Image-to-image translation

«Превратить эскиз в фото», «фото в стиль X». Через **img2img** mode SD или **ControlNet**.

### ControlNet: структурный контроль

Добавляет к диффузии condition: pose, depth, edges, segmentation map. Полный control над композицией.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
# Pass canny edges as control
image = pipe(prompt="a luxury car", image=canny_edges).images[0]
```

### LoRA для стиля или объекта

Fine-tune diffusion на 5-50 картинках, получите модель, генерящую в нужном стиле или конкретного человека.

```python
# Train своих LoRA через kohya-ss/sd-scripts или diffusers
# Подключение
pipe.load_lora_weights("my-style-lora.safetensors")
```

## Синтетические данные для CV

Generative модели могут создавать train data:

- **Augmentation для редких классов** (медицина, дефекты).
- **Domain randomization** для robotics simulation.
- **Privacy-preserving data** (synthetic faces вместо real).

**Caveat:** синтетика часто помогает, но иногда модель учится artefactsам диффузии. Eval на real data обязательна.

## Image editing с instructions

**InstructPix2Pix, MagicBrush:** редактирование через текстовую инструкцию. «Сделай фон зелёным», «добавь усы».

В 2026 топ — **FLUX.1 Tools** (Fill, Depth, Canny, Redux) или **Gemini 2.0 Native Image Gen**.

## Video generation

- **Stable Video Diffusion:** короткие клипы.
- **CogVideoX, Mochi-1:** open-source, neither SOTA but usable.
- **Sora, Veo, Kling:** closed-source SOTA.

В проде video gen ещё дорогой и редко критичен — кроме marketing/creative.

## Production constraints

### Latency

SD на 50 steps на A100 = ~3-5 sec. Optimization:
- **SDXL Turbo, SDXL Lightning** — 1-4 steps вместо 50.
- **TensorRT компиляция** — 2-3x speedup.
- **Distilled models** (DMD, LCM-LoRA).

### Cost

Self-hosted SD на A10 ≈ $0.01-0.05 за картинку.
API (Midjourney, DALL·E 3, Flux) ≈ $0.03-0.1 за картинку.

Self-host выгоден от ~10K картинок в день.

### Safety

Generative модели могут производить:
- NSFW
- Violence
- Deepfakes реальных людей
- Copyright violations

**Mitigation:**
- Safety classifier на выходе (NSFW detector, например).
- Watermarking (SynthID от Google).
- Negative prompts.
- Content filtering на промптах.

## Антипаттерны

- **Использовать SD-1.5 в 2026.** SDXL, SD3, Flux — всё лучше.
- **Игнорировать ControlNet** для precision use cases. Без control SD «сама решает».
- **Тренировать LoRA без regularization images.** Получаете overfit на единственное лицо.
- **Не делать safety eval.** Generative модели могут производить юридически проблемный контент.

## Задания

1. Использовать Stable Diffusion для генерации синтетических данных одного класса. Дообучить классификатор и сравнить с базой.
2. Реализовать background removal pipeline на 100 продуктовых фото.
3. Использовать SwinIR для super-resolution серии исторических фото. Сравнить с Real-ESRGAN.
4. Натренировать LoRA на 30 фото своего питомца через diffusers. Сгенерировать в разных стилях.
5. Использовать ControlNet (Canny) для генерации картинок по architectural sketches.
6. Реализовать «smart eraser» через inpainting: убрать объект с фото.

## Чек-лист

- [ ] Понимаю latent diffusion на пальцах.
- [ ] Умею использовать SD/Flux pipelines из diffusers.
- [ ] Знаю ControlNet, LoRA для CV.
- [ ] Понимаю практические применения: super-res, inpainting, bg removal.
- [ ] Знаю про safety и watermarking.

## Дальше

➡️ [16-ocr.md](./16-ocr.md) — OCR и распознавание документов.
