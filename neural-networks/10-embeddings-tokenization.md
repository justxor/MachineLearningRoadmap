# Урок 10. Embeddings и токенизация

> Цель урока: понять, как текст превращается в числа, на которых работает нейросеть. Освоить BPE-токенизацию и научиться использовать embeddings для поиска по смыслу (semantic search) и RAG.

## Зачем embeddings

Нейросеть не умеет работать со строками — только с числами. **Embedding** — это плотный векторный представитель токена, слова или предложения. Главное свойство: **семантически близкие объекты имеют близкие векторы**.

Классический пример (word2vec, 2013):

```
king - man + woman ≈ queen
```

Векторы захватили смысловую структуру языка просто из статистики совместной встречаемости слов.

## Токенизация: путь от текста к индексам

**Character-level.** Каждый символ — токен. Просто, словарь маленький (~100), но последовательности длинные.

**Word-level.** Каждое слово — токен. Короткие последовательности, но огромный словарь (миллионы) и проблема OOV (out-of-vocabulary).

**Subword (BPE, WordPiece, SentencePiece).** Компромисс. Самые частые комбинации байт/символов объединяются в один токен. Это **стандарт** для всех современных LLM.

## Byte Pair Encoding (BPE) с нуля

Алгоритм:

1. Начать с словаря из всех байтов/символов.
2. Найти самую частую пару соседних токенов в корпусе.
3. Слить эту пару в новый токен, добавить в словарь.
4. Повторять, пока словарь не достигнет нужного размера.

```python
from collections import Counter

def get_pairs(words):
    pairs = Counter()
    for word, freq in words.items():
        symbols = word.split()
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge(words, pair):
    new = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in words.items():
        new[word.replace(bigram, replacement)] = freq
    return new

corpus = "low low low low low lower lower newest newest newest widest widest"
words = Counter(corpus.split())
words = {' '.join(list(w)) + ' </w>': c for w, c in words.items()}

for step in range(10):
    pairs = get_pairs(words)
    if not pairs: break
    best = max(pairs, key=pairs.get)
    words = merge(words, best)
    print(f"step {step}: merged {best}")
```

После 10 итераций вы увидите, как «est», «low», «lower» становятся отдельными токенами. Реальные токенизаторы делают то же самое, только на 50k-100k итерациях.

## Готовые токенизаторы

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, World!")
print(ids)              # [15496, 11, 2159, 0]
print(tok.decode(ids))  # 'Hello, World!'
print(tok.tokenize("Hello, World!"))  # ['Hello', ',', 'ĠWorld', '!']
```

Обратите внимание: пробел перед «World» закодирован символом «Ġ» — это часть токена. Так BPE справляется с границами слов.

## Embeddings в нейросети

`nn.Embedding(vocab, dim)` — это просто **таблица** размера `vocab × dim`. По индексу токена возвращается строка. Эта таблица **обучается** вместе со всей сетью.

```python
import torch, torch.nn as nn

embed = nn.Embedding(num_embeddings=10000, embedding_dim=128)
ids = torch.tensor([42, 7, 1337])
vectors = embed(ids)   # shape (3, 128)
```

В обученной модели токены, встречающиеся в похожих контекстах, получают близкие векторы.

## Predefined embeddings: word2vec, GloVe, FastText

До эры трансформеров это были **готовые** векторные пространства слов:

- **word2vec (2013)** — обучается на задаче «предсказать слово по контексту» (CBOW) или наоборот (Skip-gram).
- **GloVe** — использует глобальную статистику совместной встречаемости.
- **FastText** — учитывает подслова, работает на редких словах.

Сейчас их редко используют напрямую, но интуиция «семантически близкие = геометрически близкие» — та же.

## Sentence/Document embeddings

Современные модели дают embedding не для слова, а для **целого текста**. Это используется в:

- **Semantic search** — найти документ по смыслу запроса.
- **RAG (Retrieval-Augmented Generation)** — LLM получает на вход релевантные документы, найденные через embeddings.
- **Кластеризация и классификация** текстов.
- **Recommendation systems** — товары/контент кодируются в общее пространство.

Готовые модели для русского и английского: `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/multilingual-e5-large`, `BAAI/bge-m3`.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = [
    "Кошка спит на диване",
    "Собака гуляет в парке",
    "Кот лежит на кровати",
    "Сегодня солнечная погода",
]
embeddings = model.encode(sentences)
sim = embeddings @ embeddings.T / (np.linalg.norm(embeddings, axis=1, keepdims=True) *
                                    np.linalg.norm(embeddings, axis=1, keepdims=True).T)
print(sim.round(2))
```

Вы увидите, что фразы про кошку/кота имеют высокую косинусную близость, а фраза про погоду — низкую ко всем.

## Векторные БД (FAISS, Qdrant, Pinecone)

Когда документов миллионы, линейный поиск становится слишком медленным. Векторные базы данных используют **ANN** (approximate nearest neighbor) — HNSW, IVF, PQ — чтобы искать за O(log n) с потерей точности 1-5%.

```python
import faiss
index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product
index.add(embeddings)
D, I = index.search(embeddings[:1], k=3)        # 3 ближайших к первому
```

## 8 практических заданий

1. **Свой BPE.** Запустите код выше. Постройте словарь до 100 токенов на корпусе из 1000 предложений (например, первая глава книги).
2. **Сравнение токенизаторов.** На одном русском предложении сравните, на сколько токенов его разбивают `gpt2`, `gpt-4o` (через tiktoken), и `xlm-roberta`. Объясните разницу.
3. **Word2vec-style.** Обучите Skip-gram word2vec на собственном корпусе текстов. Найдите ближайшие соседи для слова «кошка».
4. **Аналогии.** В вашем word2vec покажите, что `король - мужчина + женщина ≈ королева`. Если не получается — почему?
5. **Sentence embeddings.** На 1000 коротких отзывов о товаре сделайте кластеризацию через KMeans на embeddings. Опишите 5 главных кластеров.
6. **Semantic search.** Постройте поиск по 1000 статьям. Запросы вида «как обучить нейросеть» должны находить релевантные статьи, даже если в них этих слов дословно нет.
7. **FAISS.** Перенесите поиск из задания 6 на FAISS с `IndexIVFFlat`. Сравните скорость и точность с `IndexFlatIP`.
8. **Мини-RAG.** Соберите простой RAG: вопрос → embedding → top-3 документа → подаём их + вопрос в GPT-2/Mistral → ответ. Сравните с ответом без retrieval.

## Чек-лист урока

- [ ] Я могу руками описать алгоритм BPE.
- [ ] Я понимаю, чем char/word/subword токенизация отличаются.
- [ ] Я знаю, что такое `nn.Embedding` и как она обучается.
- [ ] Я делал semantic search на готовых embeddings.
- [ ] Я понимаю, что такое векторная БД и зачем она нужна.
- [ ] Я сделал хотя бы 5 заданий из 8.

---

◀︎ [Урок 09](./09-transformer.md) · [README курса](./README.md) · ▶︎ [Урок 11 — Fine-tuning](./11-finetuning.md)
