# Урок 05. Оптимизаторы: SGD, Momentum, Adam, AdamW

> Цель урока: понять, как именно нейросеть «учится» — что делает `opt.step()` за кулисами — и научиться правильно подбирать learning rate и оптимизатор под задачу.

## Базовый случай: Stochastic Gradient Descent (SGD)

$$
w_{t+1} = w_t - \eta \nabla \mathcal{L}(w_t)
$$

Просто: посчитали градиент → шагнули против него на длину `η` (learning rate). «Stochastic» — потому что считаем по мини-батчу, а не по всему датасету.

**Проблемы чистого SGD:**

- Узкие овраги loss-ландшафта → траектория зигзагом.
- Чувствителен к выбору `η`: чуть больше — расходится, чуть меньше — тащится.
- Шум от батча создаёт «прыжки» при шагах рядом с минимумом.

## SGD с моментом (Momentum)

Добавляем «инерцию» — экспоненциальное скользящее среднее градиентов:

$$
v_{t+1} = \beta v_t + (1 - \beta) \nabla \mathcal{L}(w_t), \quad w_{t+1} = w_t - \eta v_{t+1}
$$

`β` обычно `0.9`. Эффект — траектория сглаживается, в оврагах ускоряемся вдоль дна.

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

## Adam: адаптивный шаг для каждого параметра

Adam (Adaptive Moment Estimation, 2014) хранит **два** скользящих средних:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(момент 1-го порядка)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(момент 2-го порядка)}
$$

С коррекцией смещения и шагом:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

**Зачем второй момент?** Если у параметра градиент часто большой — мы автоматически делаем шаг меньше. Каждый вес получает **свой** эффективный learning rate.

**Дефолты:** `β1=0.9, β2=0.999, ε=1e-8`. Их трогают редко.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## AdamW: правильное добавление weight decay

В оригинальном Adam weight decay добавлялся к градиенту, что **смешивало** его с адаптивным шагом. AdamW (Loshchilov & Hutter, 2017) применяет weight decay **отдельно**:

$$
w_{t+1} = w_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right)
$$

На практике AdamW почти всегда лучше Adam, особенно в трансформерах. **Используйте AdamW по умолчанию.**

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

## Learning rate — главная гиперпараметрическая ручка

**Слишком большой:** loss скачет или взрывается до NaN.
**Слишком маленький:** обучение тянется вечность, может застрять в плохом локальном минимуме.

**Эвристики:**

- Adam/AdamW: `1e-3` для CNN, `3e-4` («Karpathy constant») для трансформеров, `1e-4 — 1e-5` для fine-tuning.
- SGD+Momentum: `0.1` для CNN-классики (ResNet и т.п.), затем cosine schedule.

## LR Finder — найдите свой lr за 1 эпоху

```python
import torch, math

def lr_finder(model, loader, start_lr=1e-7, end_lr=10):
    opt = torch.optim.SGD(model.parameters(), lr=start_lr)
    lrs, losses = [], []
    n = len(loader)
    mult = (end_lr / start_lr) ** (1/n)
    lr = start_lr
    for i, (x, y) in enumerate(loader):
        opt.param_groups[0]['lr'] = lr
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); opt.step()
        lrs.append(lr); losses.append(loss.item())
        lr *= mult
        if loss.item() > 4 * min(losses):
            break
    return lrs, losses
```

Постройте график `loss vs lr` (log-scale). **Оптимальный lr** — на порядок меньше точки, где loss начинает расти.

## Learning rate schedule

Хорошая практика — **уменьшать lr во время обучения**:

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
for epoch in range(100):
    train_one_epoch(...)
    scheduler.step()
```

Популярные расписания:

- **StepLR** — раз в N эпох умножить lr на γ.
- **CosineAnnealing** — плавно от `lr` до ~0.
- **OneCycleLR** — растёт от малого до большого и обратно за всё обучение. Часто работает лучше всего.
- **Warmup + cosine** — стандарт для трансформеров: 5-10% обучения lr растёт от 0 до пика, потом cosine.

## Когда какой оптимизатор использовать

| Задача | Рекомендация |
|---|---|
| CNN классификация изображений | SGD + Momentum + cosine schedule |
| Трансформеры (NLP, ViT) | AdamW + warmup + cosine |
| Fine-tuning предобученных моделей | AdamW с малым lr (`1e-5` — `5e-5`) |
| RL / нестабильные среды | Adam с малым lr |
| Sparse-данные (рекомендации, NLP с word embeddings) | Adam (адаптивность важна) |

## 8 практических заданий

1. **Свой SGD.** Реализуйте `SGDM` (SGD with momentum) как класс с методом `step()`. Используйте на MNIST. Сравните loss с `torch.optim.SGD`.
2. **Свой Adam.** Реализуйте Adam с нуля. Запустите на той же задаче. Сравните с `torch.optim.Adam` на 1000 шагах — должны совпадать с точностью 1e-5.
3. **LR Finder.** Реализуйте функцию выше, запустите на CIFAR-10 + ResNet18. Постройте график. Выберите lr.
4. **Сравнение оптимизаторов.** На CIFAR-10 обучите одну и ту же CNN с `SGD`, `SGD+Momentum`, `Adam`, `AdamW` (по 20 эпох). Сделайте таблицу test-accuracy.
5. **Расписание.** Сравните `AdamW` без schedule, с `CosineAnnealingLR` и с `OneCycleLR` на CIFAR-10. Какой даёт лучший результат?
6. **NaN-катастрофа.** Намеренно поставьте lr=10. Покажите, что loss становится NaN за 1-2 эпохи. Объясните, как защититься (`torch.nn.utils.clip_grad_norm_`).
7. **Adam vs AdamW.** Обучите трансформер из `nn.TransformerEncoder` на тексте. Сравните Adam и AdamW при одинаковом `weight_decay=0.1`. Какой даёт меньший val-loss?
8. **Warmup.** Реализуйте линейный warmup на первые 1000 шагов, потом cosine. Сравните с просто cosine.

## Чек-лист урока

- [ ] Я могу руками вывести шаг Adam.
- [ ] Я понимаю, чем AdamW отличается от Adam.
- [ ] Я умею пользоваться LR Finder.
- [ ] Я знаю, какой оптимизатор брать под каждый тип задачи.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 04](./04-regularization.md) · [README курса](./README.md) · ▶︎ [Урок 06 — CNN с нуля](./06-cnn.md)
