# 04. Pandas и NumPy на боевых данных

pandas и NumPy — основной инструмент работы с табличными данными в Python. Плохое владение превращает простые задачи в часы боли.

## Ядро

- Series, DataFrame, Index. Как устроен­ы внутри.
- Индексация: `loc`, `iloc`, булева.
- groupby + agg/transform/apply.
- merge, join, concat. Когда что уместно.
- pivot, melt, stack/unstack.
- Работа с временем: resample, rolling, expanding.
- categorical dtype и выигрыш по памяти.

## NumPy

- Векторизация: почему for-loop в 100–1000× медленнее.
- Broadcasting: правила и ловушки.
- np.where, np.select вместо if/else в apply.
- view vs copy — откуда SettingWithCopyWarning.

## Производительность

- Читайте CSV с `dtype`, `usecols`, `parse_dates`.
- `pd.read_parquet` вместо CSV для больших данных.
- Polars как альтернатива для >2 GB.
- Профилируйте через %timeit, %prun.

## Антипаттерны

- iterrows / apply вместо векторизации.
- Создание DataFrame в цикле через append (удалён в pandas 2).
- Шахматы inplace=True по всему коду.
- Не осознавать типы: object там, где должен быть category или datetime.

## Практика

1. Возьмите датасет с Kaggle (не titanic) на ~1 GB. Отпрофилируйте память.
2. Сократите память через downcast типов и category.
3. Посчитайте ролл-7 скользящее среднее на временном ряде.
4. Перепишите apply на векторизацию, измерьте выигрыш.
5. Сделайте pivot и melt туда-обратно, получите исходный DataFrame.
6. Реализуйте ABT (analytical base table) на 50+ фичах.

## Чек-лист

- [ ] Пишу векторный код.
- [ ] Понимаю разницу transform/apply/agg.
- [ ] Умею читать и писать parquet.

[← 03](03-sql.md) • [→ 05. Сбор данных](05-data-collection.md)
