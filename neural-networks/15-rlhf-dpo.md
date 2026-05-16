# Урок 15. RLHF, DPO и alignment LLM

> Цель урока: разобраться, как «сырая» предобученная LLM превращается в полезного и безопасного ассистента. Понять RLHF, PPO и более простой современный метод DPO.

## Зачем alignment

После pretraining + SFT LLM умеет много, но часто:

- Слишком формальна или слишком фамильярна — не угадывает стиль.
- Отвечает уверенно неверно (галлюцинации).
- Не отказывается от опасных запросов или, наоборот, отказывается от безобидных.
- Не следит за многоходовым диалогом.

**Alignment** — это этап обучения, на котором модель учат **соответствовать предпочтениям пользователей** (а не просто статистике корпуса).

Главный инструмент сегодня — обучение на парах сравнений: «ответ A лучше ответа B».

## Три классических этапа: SFT → Reward Model → PPO

**1. SFT (Supervised Fine-Tuning).** Дообучаем pretrained LLM на парах «инструкция → желаемый ответ». Это даёт «помощника, который старается отвечать».

**2. Reward Model (RM).** Собираем датасет пар `(prompt, response_chosen, response_rejected)`. Обучаем отдельную модель — обычно ту же архитектуру + scalar head — предсказывать «насколько ответ хороший». Loss:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

где `y_w` — выбранный (winner), `y_l` — отброшенный (loser).

**3. PPO (Proximal Policy Optimization).** LLM теперь — это политика `π_θ(y | x)`. Используем RL: генерируем ответ, RM даёт скаляр-награду, делаем шаг PPO. Чтобы политика не «убежала» от исходной (и не выучила хакать reward model), добавляем KL-штраф:

$$
\mathcal{R}(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}\!\left[\pi_\theta(\cdot | x) \,\|\, \pi_{\text{SFT}}(\cdot | x)\right]
$$

Это и есть **RLHF** (Christiano et al., 2017; Ouyang et al., InstructGPT, 2022).

**Проблема PPO:** очень нестабильно, дорого, чувствительно к гиперпараметрам. Нужно одновременно хранить 4 копии модели (policy, ref, reward, value) — память расходуется в 4 раза.

## DPO: тот же результат без RL

**DPO (Direct Preference Optimization, Rafailov et al., 2023)** заметил: можно обойтись **без** reward model и **без** PPO. Если переписать KL-регуляризованный RL аналитически, получится closed-form формула, которую можно оптимизировать как обычный supervised learning:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

То есть мы напрямую увеличиваем вероятность выбранного ответа относительно отброшенного, но **относительно** референс-модели — чтобы не уехать далеко.

**Что это даёт:**

- Нет отдельной reward model — экономия памяти и кода.
- Нет PPO, нет генерации в цикле — обучается как обычный SFT, в десятки раз стабильнее.
- При сопоставимых данных даёт качество не хуже PPO-RLHF.

**Сегодня DPO — стандарт для open-source моделей** (Zephyr, Mistral-Instruct, OpenChat и т.д.).

## Минимальный DPO-loss в PyTorch

```python
import torch, torch.nn.functional as F

def dpo_loss(model, ref_model, prompts, chosen_ids, rejected_ids, beta=0.1):
    # logprob ответов под обеими моделями
    def logp(m, prompt, response):
        ids = torch.cat([prompt, response], dim=1)
        logits = m(ids).logits[:, :-1]
        targets = ids[:, 1:]
        logp_tok = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # маскируем только токены ответа
        mask = torch.zeros_like(logp_tok)
        mask[:, prompt.size(1)-1:] = 1
        return (logp_tok * mask).sum(dim=-1)

    lp_w = logp(model, prompts, chosen_ids)
    lp_l = logp(model, prompts, rejected_ids)
    with torch.no_grad():
        ref_w = logp(ref_model, prompts, chosen_ids)
        ref_l = logp(ref_model, prompts, rejected_ids)

    logits = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -F.logsigmoid(logits).mean()
```

В библиотеке `trl` от HuggingFace это всё уже реализовано:

```python
from trl import DPOTrainer
trainer = DPOTrainer(
    model, ref_model=ref_model, beta=0.1,
    train_dataset=dataset, tokenizer=tok,
    args=TrainingArguments(...),
)
trainer.train()
```

## Современные альтернативы DPO

- **IPO (Identity Preference Optimization)** — другая параметризация loss, устойчивее к шумным меткам.
- **KTO (Kahneman-Tversky Optimization)** — обходится без пар «лучше/хуже», работает с одиночными бинарными метками «нравится/нет».
- **ORPO** — объединяет SFT и preference learning в одном шаге, не требует ref-модели.
- **GRPO** (используется в DeepSeek-R1, 2024) — групповая нормализация наград без value-функции, дешёвый PPO.

Для рабочих задач 80% случаев — DPO. Для reasoning-моделей и chain-of-thought — GRPO.

## Constitutional AI и RLAIF

В RLAIF (RL from AI Feedback) — пары сравнений генерирует не человек, а сильная LLM (GPT-4, Claude). Это в десятки раз дешевле и масштабируется. Constitutional AI (Anthropic) — частный случай: модель оценивает ответы по списку принципов («не помогай в нелегальном», «не давай медицинских советов без disclaimer» и т.д.).

## Как собрать датасет предпочтений

- **Готовые открытые:** `Anthropic/hh-rlhf`, `OpenAssistant/oasst1`, `argilla/distilabel-intel-orca-dpo-pairs`, `HuggingFaceH4/ultrafeedback_binarized`.
- **Свой:** хотя бы 1-3k пар на узкую задачу + аугментация через сильную LLM.

Чем разнообразнее prompts — тем лучше переносится. Чем строже критерии «лучше/хуже» — тем стабильнее обучение.

## 8 практических заданий

1. **Свой DPO с нуля.** Реализуйте `dpo_loss` и проверьте на синтетике (один и тот же prompt, два разных ответа, известная разметка).
2. **DPO через trl.** Возьмите Qwen-0.5B или TinyLlama. Обучите DPO на `ultrafeedback_binarized`. Сравните до/после на 20 промптах вручную.
3. **Reward model.** Обучите простую reward model на тех же данных. Сравните: как часто её предсказания совпадают с разметкой на test-сэмплах.
4. **KL-watch.** Во время DPO логируйте KL-дивергенцию `policy || ref`. Постройте график. Объясните, что значит резкий рост.
5. **Constitutional self-critique.** Возьмите свою LLM, заставьте её оценивать собственные ответы по 3 принципам, переписывайте по критике. Замерьте качество до/после на 30 промптах.
6. **KTO.** Реализуйте KTO-loss на бинарных метках «нравится/нет». Сравните с DPO при одинаковом числе примеров.
7. **GRPO для math.** Возьмите GSM8K, дообучите маленькую LLM с GRPO на правильности ответа (reward = 1 если ответ верный, 0 иначе). Замерьте accuracy до и после.
8. **Reward hacking demo.** Намеренно сделайте дырявую reward model (например, штрафует за длинные ответы). Покажите, что после PPO/DPO модель начинает отвечать односложно. Это **главная проблема** alignment.

## Чек-лист урока

- [ ] Я могу нарисовать схему «SFT → RM → PPO» и объяснить роль каждого шага.
- [ ] Я понимаю, как DPO избавляется от RM и PPO.
- [ ] Я обучил DPO-модель на готовом датасете и сравнил её ответы до/после.
- [ ] Я знаю, что такое KL-штраф и зачем он нужен.
- [ ] Я понимаю, что такое reward hacking, и могу привести пример.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 14](./14-diffusion.md) · [README курса](./README.md) · ▶︎ [Урок 16 — Графовые нейросети](./16-gnn.md)
