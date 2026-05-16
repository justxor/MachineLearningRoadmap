# Урок 16. Графовые нейросети (GNN)

> Цель урока: понять, как нейросети работают с данными, для которых нет естественного порядка — социальные сети, молекулы, графы знаний, рекомендации. Реализовать GCN и GAT с нуля.

## Где встречаются графы

- **Социальные сети** — пользователи и связи между ними.
- **Молекулы** — атомы (узлы) и химические связи (рёбра). Свойства молекулы — задача классификации/регрессии на графе.
- **Knowledge graphs** — сущности и отношения. Запрос «кто директор компании X» — link prediction.
- **Рекомендации** — двудольный граф «пользователи ↔ товары».
- **Анализ кода** — abstract syntax tree.
- **Дороги и логистика** — узлы города и связи между ними.

Что общего: **порядок узлов не определён**, но **связи между ними важны**.

## Главная идея: message passing

В каждом GNN-слое каждый узел:

1. Собирает «сообщения» от своих соседей (агрегация: sum/mean/max/attention).
2. Объединяет их со своим текущим состоянием.
3. Обновляет своё представление.

Это повторяется L раз → узел «видит» соседей на расстоянии L шагов.

Общая формула:

$$
h_v^{(l+1)} = \sigma\!\left( W^{(l)} \cdot \text{AGGREGATE}\bigl(\{ h_u^{(l)} : u \in \mathcal{N}(v) \cup \{v\} \}\bigr) \right)
$$

Конкретные GNN-архитектуры отличаются способом агрегации.

## GCN: самый простой случай

**Graph Convolutional Network (Kipf & Welling, 2017):**

$$
H^{(l+1)} = \sigma\!\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$

где `Ã = A + I` (матрица смежности с self-loops), `D̃` — степени узлов. Эта нормировка по сути усредняет фичи соседей с учётом степеней.

**Минимальная реализация:**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)

    def forward(self, X, A_norm):
        # A_norm — заранее посчитанная Ã нормированная
        return self.lin(A_norm @ X)

def normalize_adj(A):
    A_hat = A + torch.eye(A.size(0))
    D_hat = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(D_hat.pow(-0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class GCN(nn.Module):
    def __init__(self, in_f, hidden, n_classes):
        super().__init__()
        self.g1 = GCNLayer(in_f, hidden)
        self.g2 = GCNLayer(hidden, n_classes)

    def forward(self, X, A_norm):
        h = F.relu(self.g1(X, A_norm))
        return self.g2(h, A_norm)
```

Применение к классическому датасету Cora (узлы — научные статьи, рёбра — цитирования, задача — классификация по теме): GCN с 2 слоями достигает accuracy ~81%.

## GAT: attention для соседей

**Graph Attention Network (Veličković et al., 2018)** заменил равномерное усреднение на **обучаемые веса внимания** между узлами:

$$
\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^\top [W h_v \| W h_u]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}^\top [W h_v \| W h_k]))}
$$

$$
h_v' = \sigma\!\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u \right)
$$

GAT часто работает лучше GCN на гетерогенных графах, где разные соседи важны по-разному.

## Уровни задач на графах

- **Node-level.** Классификация/регрессия для каждого узла отдельно (Cora, citation classification).
- **Edge-level (link prediction).** Предсказать, есть ли связь между двумя узлами. Применение: рекомендации, дополнение knowledge graph.
- **Graph-level.** Свойство всего графа. Применение: предсказание свойств молекул, классификация программ.

Для graph-level задач нужен **readout** — агрегация всех узлов в один вектор: `mean / sum / max / attention pooling`.

## PyTorch Geometric — стандарт де-факто

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv

dataset = Planetoid(root='./Cora', name='Cora')
data = dataset[0]

import torch.nn as nn, torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, training=self.training)
        return self.conv2(h, edge_index)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(200):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
```

## Современный landscape

- **GraphSAGE** — масштабируется на огромные графы через семплирование соседей.
- **GIN (Graph Isomorphism Network)** — теоретически наиболее выразительная среди message-passing GNN.
- **Graph Transformers** — переносят attention на всю «глобальную» структуру графа. SOTA на многих бенчмарках 2024-2025.
- **Equivariant GNN (E(n)-GNN, EGNN)** — учитывают геометрию (важно для молекул, физических симуляций). На них построен **AlphaFold**.

## Когда НЕ использовать GNN

- Если граф можно превратить в табличку без потерь — XGBoost обычно лучше.
- Если структура простая и почти полносвязная — обычный трансформер с positional encoding часто выигрывает.
- Если меток мало (<100 узлов) — GNN переобучится, лучше label propagation или классические графовые алгоритмы.

## 8 практических заданий

1. **GCN с нуля.** Реализуйте `GCN` без `torch_geometric`. Запустите на Cora через прямую матрицу смежности. Получите test-accuracy > 78%.
2. **GAT с нуля.** Реализуйте `GATLayer` с 4 головами. Обучите на Cora. Сравните с GCN.
3. **Над- и подъглаживание (over-smoothing).** Постройте GCN с 2, 4, 8, 16 слоями. Покажите, что accuracy падает с глубиной. Это **главная боль** GNN.
4. **Link prediction.** На Cora добавьте «отрицательные» рёбра (рандомные несвязанные пары). Обучите модель отличать настоящие рёбра от случайных. AUC > 0.85.
5. **MUTAG (graph-level).** Загрузите `TUDataset(name='MUTAG')` — молекулы и метка мутагенности. Обучите GIN с mean pooling. Получите test-accuracy > 80%.
6. **GraphSAGE на большом графе.** Возьмите `OGB ogbn-arxiv` (170k узлов). Обучите GraphSAGE с neighbor sampling. Замерьте время эпохи.
7. **Embeddings узлов.** На Cora обучите Node2Vec (через `torch_geometric.nn.Node2Vec`). Визуализируйте 2D эмбеддинги через t-SNE. Кластеры должны совпадать с темами статей.
8. **GNN для рекомендаций.** На MovieLens-100k постройте двудольный граф user-item. Обучите LightGCN. Сравните `recall@10` с baseline на матричной факторизации.

## Чек-лист урока

- [ ] Я могу руками вывести шаг GCN-слоя.
- [ ] Я понимаю, чем GAT отличается от GCN.
- [ ] Я обучил GNN на Cora и достиг разумной accuracy.
- [ ] Я знаю, что такое over-smoothing и почему GNN редко делают глубокими.
- [ ] Я понимаю разницу между node-level, edge-level и graph-level задачами.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 15](./15-rlhf-dpo.md) · [README курса](./README.md) · ▶︎ [Урок 17 — Self-supervised learning](./17-self-supervised.md)
