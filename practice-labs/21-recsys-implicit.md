# 🎬 Лаба 21: Collaborative filtering на implicit feedback 🟡

## Цель

Построить рекомендательную систему на неявных сигналах (просмотры, клики, продажи). Научиться правильно строить eval-протокол и избегать data leakage.

## Датасет

- MovieLens 1M (implicit вариант: rating>=4 → 1).
- Last.FM listening history.
- Retail Rocket events.

## Минимальный пайплайн

1. Сборка user-item matrix.
2. Leave-last-out split (по времени) + negative sampling для eval.
3. Baselines: Popular, Item-kNN.
4. ALS (implicit library) или BPR.
5. Оценка: HitRate@10, NDCG@10, Coverage, Novelty.
6. Сравнение моделей в таблице.

## Код: ALS (implicit)

```python
import implicit, scipy.sparse as sp

ui = sp.csr_matrix((data, (users, items)))
model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
model.fit(ui)

recommendations = model.recommend(user_id=42, user_items=ui[42], N=10)
```

## Метрики

- HitRate@K, Recall@K, Precision@K.
- NDCG@K (учёт порядка).
- MAP@K.
- Coverage — доля товаров в рекомендациях.
- Novelty — сколько не «популярных».
- Diversity (intra-list).

## Расширения

- LightFM (гибрид: CF + content).
- BERT4Rec / SASRec для sequence-aware.
- Two-tower neural CF.
- Cold start: content-based fallback.
- A/B-дизайн для deployment.

## Критерии приёмки

- [ ] Time-based split (не random).
- [ ] ALS побеждает Popular по NDCG на ≥5%.
- [ ] Coverage и Novelty в таблице.
- [ ] Инференс для user_id выдаёт top-K за <100мс.
- [ ] Кейсы cold start разобраны.

## Анти-паттерны

- ❌ Random split вместо time-based (leakage из будущего).
- ❌ Оценка на всех товарах без negative sampling (нереалистично).
- ❌ Игнор coverage — все пользователи получают одно и то же.
- ❌ Popular как baseline выигрывает — значит модель не учится.

---

[← Назад к Practice Labs](./README.md)
