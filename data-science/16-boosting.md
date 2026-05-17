# 16. Бустинги: XGBoost, LightGBM, CatBoost

Бустинги — рабочая лошадь в табличных задачах. Для большинства прикладных кейсов это правильный первый выбор после бейзлайна.

## Быстрое сравнение

- **XGBoost.** Стандарт по умолчанию, хорошая документация.
- **LightGBM.** Быстрее на больших данных, leaf-wise.
- **CatBoost.** Лучше работает с категориями из коробки.

## Ключевые гиперпараметры

- learning_rate (eta) и n_estimators — баланс.
- max_depth / num_leaves.
- min_child_weight / min_data_in_leaf.
- subsample, colsample_bytree.
- L1/L2 regularization (reg_alpha, reg_lambda).
- early_stopping_rounds.

## Пример

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8,
    reg_alpha=0.1, reg_lambda=0.1
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50)])
```

## Подбор гиперпараметров

- Optuna с TPE.
- Cross-validation в objective.
- Сначала «грубые» параметры (глубина, lr), потом регуляризация.
- Не тратьте время на микротюнинг пока фичи не хороши.

## Интерпретация

- feature_importances_ — быстрый взгляд, бывает лжив.
- Permutation importance — надёжнее.
- SHAP — локальные + глобальные объяснения.

## Антипаттерны

- Бустинг без early stopping.
- Обучать на всех фичах без отбора, потом ругать модель за то, что «всё медленно».
- Игнорировать дисбаланс (scale_pos_weight, class_weight).

## Практика

1. Обучите LightGBM и CatBoost на одном датасете. Сравните.
2. Настройте Optuna с 50 траялами.
3. Постройте SHAP-водопад.
4. Сравните importance и permutation_importance.
5. Попробуйте разные способы обработки категорий.
6. Сохраните модель в onnx / native format.

## Чек-лист

- [ ] Использую early stopping.
- [ ] Понимаю разницу leaf-wise / level-wise.
- [ ] Подбираю гиперпараметры честно.

[← 15](15-validation.md) • [→ 17. Интерпретация](17-interpretation.md)
