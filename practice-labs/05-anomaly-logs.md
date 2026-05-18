# 🚨 Лаба 05: Anomaly detection в логах 🔴

## Цель

Обнаруживать аномалии в логах веб-сервиса (HTTP, error rate, latency). Научиться работать без размеченных данных и выбирать правильные алгоритмы.

## Датасет

- KDDCup99, NSL-KDD или Numenta Anomaly Benchmark.
- Альтернатива: синтетические логи с внедрёнными аномалиями.

## Минимальный пайплайн

1. Парсинг логов и агрегация по окнам (1 мин).
2. Фичи: частоты кодов, перцентили latency, unique IPs.
3. Сравнение: IsolationForest, LOF, One-Class SVM, Autoencoder.
4. Алертинг с порогом по P95 ошибки реконструкции.
5. Time-aware валидация.

## Код: Autoencoder

```python
import torch.nn as nn
class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8))
        self.dec = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self,x): return self.dec(self.enc(x))
```

## Метрики

- Precision@K на размеченных инцидентах (если есть).
- F1 при разных порогах.
- Mean Time To Detect (MTTD) — ключевая бизнес-метрика.
- False positive rate на «чистых» днях.

## Расширения

- Seasonal decomposition + residual anomaly.
- LSTM-Autoencoder для последовательностей.
- Online detection (скользящее окно).
- Grouping алертов (не спамить).
- Интеграция с Prometheus/Grafana.

## Критерии приёмки

- [ ] Сравнение 4 алгоритмов в таблице.
- [ ] Графики аномалий во времени.
- [ ] FPR <5% на нормальных данных.
- [ ] Runbook: что делать при алерте.

## Анти-паттерны

- ❌ Обучение AE на данных, включающих аномалии (модель «привыкает»).
- ❌ Фиксированный порог без пересмотра.
- ❌ Спам алертов без grouping/throttling.
- ❌ Отсутствие runbook (оператор не знает что делать).

---

[← Назад к Practice Labs](./README.md)
