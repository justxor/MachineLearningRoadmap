# 📉 Лаба 02: Customer churn с интерпретируемостью 🟡

## Цель

Предсказать отток клиентов и объяснить каждый предикт бизнесу. Свяжешь вероятность оттока с финансовым эффектом от удержания.

## Датасет

- Telco Customer Churn (Kaggle) — ~7000 клиентов, 21 признак.
- Таргет: Churn (Yes/No).

## Минимальный пайплайн

1. EDA: баланс классов, распределения по тарифам и сроку.
2. Stratified split 70/15/15.
3. Baseline: LogisticRegression → ROC-AUC.
4. Основная модель: LightGBM с class_weight="balanced".
5. SHAP: global feature importance + 5 индивидуальных объяснений.
6. Порог отбирается по F1 на валидации, не по дефолтным 0.5.

## Код: SHAP-объяснения

```python
import shap, lightgbm as lgb

model = lgb.LGBMClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_val)
shap.summary_plot(sv, X_val)             # глобальный
shap.force_plot(explainer.expected_value, sv[42], X_val.iloc[42])  # локальный
```

## Метрики и baseline

- ROC-AUC, PR-AUC — обе обязательно (дисбаланс классов).
- F1 при оптимальном пороге.
- Lift@10% — сколько «уходящих» попадёт в топ-10% по вероятности.
- Baseline LR: AUC ≈0.83, LightGBM: AUC ≈0.86+.

## Расширения

- Монетизация: связать P(churn) с LTV и показать expected revenue saved.
- Calibration plot (sklearn.calibration) + isotonic regression.
- Counterfactual explanations: что изменить, чтобы снизить риск.
- A/B-дизайн ретеншн-кампании на топ-10% рисковых.

## Критерии приёмки

- [ ] Скрипт обучения и инференса вынесены в .py.
- [ ] SHAP summary + 5 локальных объяснений в отчёте.
- [ ] Порог подобран и зафиксирован.
- [ ] Приведён бизнес-вывод: сколько денег сэкономит модель.
- [ ] Calibration plot в отчёте.

## Анти-паттерны

- ❌ Accuracy как единственная метрика.
- ❌ Слепое использование SMOTE без проверки на valid.
- ❌ Интерпретация важностей по gain без учёта корреляций.
- ❌ Игнорирование временной природы (утечка из будущего).

---

[← Назад к Practice Labs](./README.md)
