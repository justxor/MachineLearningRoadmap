# 👯 Лаба 22: Hybrid recsys с two-tower моделью 🔴

## Цель

Построить neural retrieval модель (two-tower) для рекомендаций большого каталога (>100k items). Научиться выбирать negative sampling и быстрый ANN индекс.

## Архитектура

- User tower: user_id embedding + side features (возраст, история) → dense → эмбеддинг.
- Item tower: item_id + content features → dense → эмбеддинг.
- Loss: in-batch sampled softmax / contrastive.
- Inference: предвычислить item эмбеддинги → FAISS/ScaNN.

## Минимальный пайплайн

1. Подготовка фичей для users и items.
2. Dataset поситивов (взаимодействия).
3. Обучение two-tower (TF Recommenders / PyTorch).
4. Sampled softmax loss.
5. Построение FAISS индекса по item эмбеддингам.
6. Eval: HitRate/NDCG на leave-last-out.
7. Latency поиска (должно быть <10мс для 1M items).

## Код: PyTorch two-tower

```python
import torch, torch.nn as nn

class Tower(nn.Module):
    def __init__(self, n_emb, n_cont, dim=64):
        super().__init__()
        self.emb = nn.Embedding(n_emb, 32)
        self.cont = nn.Linear(n_cont, 32)
        self.out = nn.Sequential(nn.Linear(64, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, ids, cont):
        return self.out(torch.cat([self.emb(ids), self.cont(cont)], dim=-1))
```

## Метрики

- HitRate@K, NDCG@K.
- Recall@K (retrieval quality).
- Latency p99 inference.
- Cold-start performance (новые юзеры и items).
- Embedding visualization (UMAP).

## Расширения

- Sequence-aware tower (user_history → BERT4Rec).
- Multi-task: обучать одновременно click и purchase.
- Reranker после retrieval (LightGBM по top-100).
- Online learning (обновление user embedding в real-time).
- Embedding quantization для экономии памяти.

## Критерии приёмки

- [ ] Two-tower обучается, loss падает.
- [ ] FAISS индекс построен.
- [ ] HitRate@10 лучше ALS-baseline.
- [ ] Latency retrieval p99 <10мс.
- [ ] Cold-start tested для новых users/items.
- [ ] Embedding visualization (кластеры видны).

## Анти-паттерны

- ❌ Full softmax по всем items (медленно для больших каталогов).
- ❌ Random negatives вместо in-batch (плохие hard negatives).
- ❌ Одинаковые эмбеддинги для всех без регуляризации — collapse.
- ❌ Игнор cold start (новые items невидимы).

---

[← Назад к Practice Labs](./README.md)
