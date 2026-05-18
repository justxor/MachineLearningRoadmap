# 🤖 Лаба 14: LLM-агент с инструментами и памятью 🔴

## Цель

Построить агента, который решает многошаговые задачи: веб-поиск, вычисления, работа с файлами, API. Научиться дизайну tool-calling, planner-executor архитектуре и обработке ошибок.

## Сценарии 

- Travel planner: найти билеты + отель + сделать маршрут.
- Data analyst: взять CSV → анализ → презентация.
- Code assistant: рефактор + тесты на репо.

## Минимальный пайплайн

1. Tool registry: 4–6 инструментов с JSON Schema.
2. ReAct или planner-executor луп.
3. Short-term memory (scratchpad) + long-term (vector store).
4. Error handling: retry, fallback.
5. Trace-логи в файл/LangSmith.
6. UI: streamlit с визуализацией шагов.

## Код: tool calling (OpenAI)

```python
from openai import OpenAI
client = OpenAI()

tools = [{"type":"function","function":{"name":"search_web","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]

def run(messages):
    for _ in range(10):
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        m = r.choices[0].message
        if not m.tool_calls: return m.content
        for tc in m.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments)
            messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
```

## Метрики

- Success rate на наборе из 30 задач.
- Average steps to solve.
- Tool call accuracy (правильные аргументы).
- Cost per task ($).
- Latency p95.

## Расширения

- Multi-agent: planner + N executors.
- Self-reflection: агент проверяет свои результаты.
- Code execution в sandbox (E2B, Daytona).
- Caching инструментов.
- Guardrails: проверка опасных вызовов (file delete, send money).

## Критерии приёмки

- [ ] Success rate >70% на бенчмарке 30 задач.
- [ ] Trace-логи доступны для каждого рана.
- [ ] Обработка ошибок (не виснет на timeout).
- [ ] Cost и latency измерены.
- [ ] Guardrails: проверка на prompt injection.

## Анти-паттерны

- ❌ Бесконечный цикл без max_steps.
- ❌ Агент «выполняет» опасные команды из промпта (промпт инъекция).
- ❌ Нет логирования — невозможно дебажить.
- ❌ Игнорирование стоимости (бюджет вылетит в трубу).

---

[← Назад к Practice Labs](./README.md)
