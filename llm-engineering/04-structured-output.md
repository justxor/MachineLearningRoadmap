# 04. Структурированный вывод: JSON mode, schemas, function calling

90% LLM-приложений в проде нуждаются не в свободном тексте, а в структурированных данных, которые можно скормить дальше в код. "Просим вернуть JSON и парсим" — это путь к багам и тревожным алертам в 3 утра.

## Три уровня надёжности

1. **Просто просим в промпте**: "верни JSON в формате {...}". Работает в 80–95% случаев, ломается на длинных ответах и edge-case'ах.
2. **JSON mode**: провайдер гарантирует валидный JSON, но не его структуру.
3. **Structured outputs / JSON schema / function calling**: гарантия валидности И соответствия схеме.

В проде используйте уровень 3 везде, где он доступен.

## OpenAI Structured Outputs

```python
from pydantic import BaseModel
from openai import OpenAI

class Contract(BaseModel):
    date: str
    parties: list[str]
    amount: float

response = OpenAI().beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=Contract,
)
contract: Contract = response.choices[0].message.parsed
```

Гарантия: `contract` — валидный объект Pydantic. Никаких `try/except JSONDecodeError`.

## Anthropic: tool use для структурированного вывода

У Anthropic нет прямого аналога Structured Outputs, но есть надёжный паттерн: объявляете "инструмент" с JSON schema и форсируете её использование.

```python
tools = [{
    "name": "extract_contract",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "parties": {"type": "array", "items": {"type": "string"}},
            "amount": {"type": "number"}
        },
        "required": ["date", "parties", "amount"]
    }
}]
```

## Pydantic + instructor

Универсальная библиотека для нескольких провайдеров с автоматическим ретраем при невалидной схеме:

```python
import instructor
client = instructor.from_openai(OpenAI())
contract = client.chat.completions.create(
    model="gpt-4o", response_model=Contract, messages=[...]
)
```

## Function calling: основа агентов

Function calling — это структурированный вывод "для вызова функции". Модель решает, какую функцию из списка вызвать и с какими аргументами. Подробнее в уроке про tools.

## Антипаттерны

- Парсить ответ модели регулярками. Это всегда ломается на edge case.
- Просить "верни Python dict" вместо JSON. Получите код, который надо eval'ить — а это RCE.
- Сложные вложенные схемы из 20 полей. Модель будет ошибаться. Разбейте на несколько вызовов.
- Не валидировать вывод даже после structured outputs. Бывают баги провайдеров.

## Практика

1. Возьмите задачу извлечения сущностей. Сделайте три версии: голый промпт, JSON mode, structured outputs. Сравните % валидных ответов на 100 примерах.
2. Опишите Pydantic-модель из 5 полей с enum и Optional. Прогоните через structured outputs.
3. Реализуйте универсальный экстрактор: на вход — Pydantic-модель и текст, на выход — заполненный объект. Поддержите OpenAI и Anthropic.
4. Сделайте схему с вложенным списком объектов (например, список позиций счёта). Проверьте, как модель справляется с длинными списками.
5. Добавьте валидацию через Pydantic validators (например, формат даты). Поймайте случаи, где модель выдаёт валидный JSON, но невалидные данные.
6. Сравните стоимость и latency между голым промптом и structured output. Зафиксируйте.

## Чек-лист

- [ ] Не парсю текстовый вывод LLM регулярками.
- [ ] Использую Pydantic / JSON schema для всех структурированных задач.
- [ ] Знаю, какой механизм поддерживает мой провайдер.
- [ ] Валидирую вывод даже после structured outputs.
- [ ] Сложные схемы декомпозирую на несколько вызовов.

---

← [Назад: промпт-инжиниринг](./03-prompt-engineering.md) | [Дальше: управление контекстом →](./05-context-management.md)
