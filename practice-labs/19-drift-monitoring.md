# 📊 Лаба 19: Мониторинг дрейфа модели в проде 🔴

## Цель

Отслеживать data drift, prediction drift, concept drift в реальном времени. Настроить алерты и dashboards. Различать разные виды дрейфа и реагировать правильно.

## Стек

- Evidently AI / NannyML / WhyLabs / Aporia.
- Prometheus + Grafana для metric storage.
- Custom statistical tests: KS, PSI, Wasserstein.

## Минимальный пайплайн

1. Сохранение production inputs/outputs (с sampling если много).
2. Reference dataset (train + recent valid).
3. Job по расписанию: сравнение распределений.
4. Evidently report: data_drift_dataset.
5. Публикация в Grafana dashboard.
6. Alert rules: PSI >0.25 (warning), >0.5 (critical).

## Код: Evidently drift report

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=ref_df, current_data=cur_df)
report.save_html("drift_report.html")
result = report.as_dict()
for m in result["metrics"]:
    print(m["result"].get("dataset_drift"))
```

## Метрики

- PSI (Population Stability Index) per feature.
- KS-statistic + p-value.
- Wasserstein distance.
- Prediction distribution shift.
- Target drift (если есть labels).
- Performance metrics decay.

## Различия видов дрейфа

- **Data drift** — входы изменились. Реакция: переобучить.
- **Concept drift** — P(y|x) изменилась. Реакция: нужны новые лейблы.
- **Prediction drift** — выходы изменились. Может быть следствием любого из выше.
- **Label/feature delay** — лябелы ещё не пришли; мониторьте прокси.

## Расширения

- NannyML: предсказание падения метрики без лейблов (CBPE/DLE).
- Slice-based monitoring (по сегментам).
- A/B между версиями модели.
- Auto-rollback при alert.
- Cost-aware alerts (не переобучать в пик).

## Критерии приёмки

- [ ] Reference и current datasets сохраняются.
- [ ] Drift report генерируется ежедневно.
- [ ] Grafana dashboard с ключевыми метриками.
- [ ] Alert rules настроены в PagerDuty/Slack.
- [ ] Runbook: что делать при каждом типе алерта.
- [ ] Симуляция дрейфа: инъектируется shift в фичу, алерт срабатывает.

## Анти-паттерны

- ❌ Мониторить только accuracy — поздно реагируете.
- ❌ PSI сравнивает разные binnings (бессмысленно).
- ❌ Reference dataset = train форевер (устаревает).
- ❌ Много false positives — операторы игнорируют алерты.
- ❌ Нет runbook — неясно что делать.

---

[← Назад к Practice Labs](./README.md)
