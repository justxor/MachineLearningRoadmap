# 💳 Лаба 03: Credit scoring под смещение классов 🟡

## Цель

Построить модель оценки кредитного риска на сильно несбалансированных данных. Научиться выбирать правильные метрики и работать с дисбалансом классов.

## Датасет

- Home Credit Default Risk (Kaggle) или Give Me Some Credit.
- Доля положительного класса: 6–8%.

## Минимальный пайплайн

1. EDA с акцентом на баланс классов.
2. Stratified k-fold валидация.
3. LightGBM с взвешенными классами.
4. Сравнение: без весов, с class_weight, с SMOTE, с focal loss.
5. Калибровка вероятностей (isotonic/sigmoid).
6. Анализ на fairness: как предикты отличаются по возрасту/полу.

## Метрики и baseline

- ROC-AUC (главная в кредитовании), KS-statistic.
- PR-AUC для редкого класса.
- Gini coefficient = 2*AUC-1, цель >0.4.
- Expected calibration error (ECE) <0.05.

## Расширения

- Stacking: LR + LightGBM + CatBoost.
- Out-of-time валидация (берём последние 3 месяца как test).
- Feature selection через Boruta / null importances.
- Permutation importance вместо gain importance.
- Reject inference: моделируем отказников.

## Критерии приёмки

- [ ] AUC на hold-out ≥0.78.
- [ ] Calibration plot в отчёте.
- [ ] Сравнение 4 способов работы с дисбалансом в таблице.
- [ ] Бизнес-вывод: прибыль/убыток при разных порогах.
- [ ] Fairness-аудит по двум группам.

## Анти-паттерны

- ❌ Оценка по accuracy при дисбалансе 1:15.
- ❌ SMOTE на всём датасете до split (leakage).
- ❌ Игнорирование калибровки (при взвешивании вероятности врут).
- ❌ Обучение на всех данных без OOT-валидации (модель развалится в проде).

---

[← Назад к Practice Labs](./README.md)
