# Урок 18. Распределённое обучение и масштабирование

> Цель урока: понять, как обучают модели, которые не помещаются в одну GPU. Освоить DDP, FSDP, mixed precision и gradient checkpointing — стандартный стек обучения LLM и больших ViT.

## Когда нужно распределённое обучение

- Модель не помещается в память одной GPU (Mistral 7B в fp16 — 14 ГБ, требует A100/H100; 70B — 140 ГБ, никуда не помещается).
- Обучение на одной GPU слишком долгое — хочется параллелить по нескольким.
- Датасет физически не влезает на один сервер.

## Три измерения параллелизма

**1. Data Parallelism (DDP).** Каждая GPU имеет **полную копию модели**, но обрабатывает свою часть батча. После backward градиенты усредняются между GPU через `all-reduce`. Так работает большинство обучений мелких/средних моделей.

**2. Model Parallelism / Tensor Parallelism (TP).** Сама модель режется по нескольким GPU — например, матрица весов `Linear` разбивается на колонки. Каждая GPU считает свою часть, потом результаты собираются. Стандарт: Megatron-LM. Подходит для очень больших слоёв.

**3. Pipeline Parallelism (PP).** Разные слои модели лежат на разных GPU. Батч режется на «микробатчи», которые текут по конвейеру. GPipe, PipeDream.

В реальности крупные LLM используют **3D-параллелизм** = DDP × TP × PP. Для нас 90% случаев — это DDP и FSDP.

## DDP в PyTorch — практика

```python
import os, torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train():
    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    model = MyModel().to(device)
    model = DDP(model, device_ids=[device])

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(N):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step(); opt.zero_grad()
    cleanup()

# Запуск:
# torchrun --nproc_per_node=8 train.py
```

`DistributedSampler` гарантирует, что каждая GPU видит свою часть датасета. `DDP` автоматически делает all-reduce на градиентах.

## FSDP: когда модель не помещается даже в одну GPU

**FSDP (Fully Sharded Data Parallel)** — это эволюция DDP, в которой **сами веса**, а не только градиенты, шардируются между GPU. Каждая GPU хранит только `1/N` параметров. Перед forward слой собирается из шардов через `all-gather`, после — снова разбирается.

Идея пришла из **ZeRO** (Microsoft DeepSpeed):

- **ZeRO-1** — шардирование состояний оптимизатора.
- **ZeRO-2** — + шардирование градиентов.
- **ZeRO-3** — + шардирование весов (= FSDP).

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
model = FSDP(model, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

С FSDP + bfloat16 + activation checkpointing можно обучать Llama-7B на 4× A100 80GB или Llama-13B на 8× A100 40GB.

## Mixed precision: fp16 / bf16

Полная точность — `fp32` (4 байта на параметр). На современных GPU вычисления в `fp16` или `bf16` (2 байта) идут **в 2-4 раза быстрее** и требуют в 2 раза меньше памяти.

**fp16** — узкий диапазон, может переполняться. Решение — **loss scaling**: масштабируем loss перед backward, делим градиенты обратно.

**bf16** — тот же диапазон, что и fp32, но меньше точность мантиссы. Loss scaling не нужен. **Используйте bf16 по умолчанию** на A100/H100.

```python
scaler = torch.cuda.amp.GradScaler()       # для fp16
for x, y in loader:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    opt.zero_grad()
```

## Gradient accumulation: большой effective batch на одной GPU

Если хочется батч 512, но в память влезает только 32 — делаем 16 шагов forward+backward без `opt.step()`, потом один `step()`:

```python
ACCUM = 16
opt.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = F.cross_entropy(model(x), y) / ACCUM
    loss.backward()
    if (i + 1) % ACCUM == 0:
        opt.step(); opt.zero_grad()
```

Эффективный батч = `micro_batch × ACCUM × world_size`.

## Gradient checkpointing: время за память

При обычном backward все активации forward хранятся в памяти. Для глубоких моделей это много гигабайт.

**Gradient checkpointing** хранит только activation на границах блоков, остальное **пересчитывает** заново при backward. Память падает в ~√N раз, время растёт на ~30%. Часто это единственный способ обучать большие модели.

```python
from torch.utils.checkpoint import checkpoint
def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

## Готовые инструменты

Не пишите свой fsdp-loop. Используйте:

- **🤗 Accelerate** — обёртка над DDP/FSDP/DeepSpeed. Один и тот же код запускается на 1 GPU, 8 GPU, мультиноды.
- **DeepSpeed** — Microsoft. ZeRO, pipeline, MoE. Хорош для очень больших моделей.
- **PyTorch Lightning + Fabric** — высокоуровневая обёртка над тренировочным циклом.
- **Trainer от HuggingFace** — закрывает 80% сценариев fine-tuning из коробки.

```python
from accelerate import Accelerator
acc = Accelerator(mixed_precision='bf16')
model, opt, loader = acc.prepare(model, opt, loader)
for x, y in loader:
    loss = F.cross_entropy(model(x), y)
    acc.backward(loss); opt.step(); opt.zero_grad()
```

Этот код работает и на 1, и на 100 GPU.

## Грабли распределённого обучения

- **Не-детерминированность.** При DDP+nondeterministic операциях два запуска могут разойтись. Для воспроизводимости — `torch.use_deterministic_algorithms(True)` (медленнее).
- **Утечки памяти.** Не накапливайте loss/predictions в питон-листах из forward — используйте `.detach().cpu()` или `torchmetrics`.
- **Дисбаланс батчей.** Если последний батч на одной GPU меньше — `all-reduce` ждёт всех. Решение — `drop_last=True` в DataLoader.
- **NCCL hangs.** Если что-то падает на одной GPU, остальные молча зависнут. Всегда логируйте `local_rank` и используйте `NCCL_DEBUG=INFO` при отладке.

## Как замерять и оптимизировать

- **`nvidia-smi`** — занятость GPU. Если <90% — узкое место, скорее всего, в DataLoader.
- **`torch.profiler`** — детальный breakdown по операциям.
- **`memory_summary()`** — где течёт память.
- **Throughput метрика для LLM** — tokens/sec/GPU. Стандарт сравнения.

## 8 практических заданий

1. **Свой DDP.** Перепишите тренировочный цикл из урока 06 (CNN на CIFAR-10) под DDP. Запустите на 2 GPU через `torchrun`. Сравните время эпохи с однопроцессорным.
2. **bf16 vs fp32.** На том же CNN сравните точность и время эпохи в fp32 и bf16. Замерьте максимально вмещаемый батч.
3. **Gradient accumulation.** Достигните effective batch = 1024 на маленькой GPU через accumulation. Сравните финальную accuracy с прямым batch=1024 на большой GPU.
4. **Gradient checkpointing.** Возьмите ViT-base, включите checkpointing. Замерьте: пиковая память, время эпохи. Сравните без checkpointing.
5. **FSDP для Mistral.** Через `accelerate` запустите FSDP fine-tuning Mistral-7B на 2-4 GPU (даже на T4 в Colab Pro+). Покажите, что модель загружается полностью.
6. **DeepSpeed ZeRO-3.** То же самое через `deepspeed` с config-файлом. Сравните throughput.
7. **Профилирование.** Запустите `torch.profiler` на одном шаге обучения. Найдите узкое место (data loading? forward? backward? optimizer?).
8. **Свой LR scaling.** При увеличении world_size в 8 раз правильно ли остаётся lr тем же? Реализуйте linear scaling rule + warmup. Замерьте разницу в final loss.

## Чек-лист урока

- [ ] Я могу написать DDP-цикл с нуля.
- [ ] Я понимаю разницу между DDP, FSDP и DeepSpeed ZeRO.
- [ ] Я знаю, чем bf16 лучше fp16 на современных GPU.
- [ ] Я умею комбинировать gradient accumulation и checkpointing.
- [ ] Я понимаю, что такое 3D-параллелизм и когда он нужен.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 17](./17-self-supervised.md) · [README курса](./README.md) · 🏠 [Главный README](../README.md)
