# 📝 NLP Snippets — Обработка Текста

Готовые сниппеты для NLP задач: от базовой обработки текста до трансформеров.

---

## 1. Предобработка текста

```python
import re
import string
import unicodedata
import pandas as pd
import numpy as np

# Установка: pip install nltk spacy pymorphy2
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

def preprocess_text_pipeline(texts: list, language: str = 'russian',
                               do_lower: bool = True,
                               do_remove_punct: bool = True,
                               do_remove_stopwords: bool = True,
                               do_stem: bool = False,
                               min_token_len: int = 2) -> list:
    """Полный пайплайн предобработки текста"""
    if language == 'russian':
        stop_words = set(stopwords.words('russian'))
        stemmer = SnowballStemmer('russian')
    else:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()

    processed = []
    for text in texts:
        if not isinstance(text, str):
            processed.append('')
            continue

        # Unicode нормализация
        text = unicodedata.normalize('NFKC', text)

        # Удаление URL, email, цифр
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        text = re.sub(r'\S+@\S+\.\S+', ' EMAIL ', text)
        text = re.sub(r'\d+', ' NUM ', text)

        if do_lower:
            text = text.lower()

        if do_remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'[^\w\s]', ' ', text)

        # Токенизация
        tokens = word_tokenize(text, language='russian' if language == 'russian' else 'english')

        if do_remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]

        tokens = [t for t in tokens if len(t) >= min_token_len]

        if do_stem:
            tokens = [stemmer.stem(t) for t in tokens]

        processed.append(' '.join(tokens))

    return processed
```

---

## 2. TF-IDF Векторизация

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
import scipy.sparse as sp

def build_tfidf_features(texts_train: list, texts_test: list,
                          max_features: int = 10000,
                          ngram_range: tuple = (1, 2),
                          use_svd: bool = False,
                          n_components: int = 100) -> tuple:
    """TF-IDF + опциональная SVD (LSA)"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,  # log(1+tf) вместо tf
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}',
    )

    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    print(f"TF-IDF matrix: {X_train.shape}")
    print(f"Sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.3f}")

    if use_svd:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        print(f"After SVD: {X_train.shape}")
        print(f"Explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    return X_train, X_test, vectorizer
```

---

## 3. Word2Vec / FastText эмбеддинги

```python
from gensim.models import Word2Vec, FastText
import numpy as np

def train_word2vec(tokenized_texts: list, vector_size: int = 100,
                   window: int = 5, min_count: int = 2) -> Word2Vec:
    """Обучение Word2Vec на своих данных"""
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-Gram (1) или CBOW (0)
        negative=10,
        epochs=10,
        seed=42,
    )
    print(f"Vocabulary size: {len(model.wv)}")
    return model

def text_to_vector(text: str, model: Word2Vec,
                   strategy: str = 'mean') -> np.ndarray:
    """Преобразование текста в вектор"""
    tokens = text.split()
    vectors = [model.wv[t] for t in tokens if t in model.wv]

    if not vectors:
        return np.zeros(model.vector_size)

    if strategy == 'mean':
        return np.mean(vectors, axis=0)
    elif strategy == 'max':
        return np.max(vectors, axis=0)
    elif strategy == 'sum':
        return np.sum(vectors, axis=0)

def texts_to_matrix(texts: list, model: Word2Vec,
                     strategy: str = 'mean') -> np.ndarray:
    """Матрица эмбеддингов для списка текстов"""
    return np.array([text_to_vector(text, model, strategy) for text in texts])

# Загрузка предобученных векторов (русский):
# from gensim.models import KeyedVectors
# wv = KeyedVectors.load_word2vec_format('cc.ru.300.vec', binary=False, limit=500000)
```

---

## 4. BERT / Transformers эмбеддинги

```python
# pip install transformers torch sentence-transformers
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

def get_sentence_embeddings(texts: list, model_name: str = 'all-MiniLM-L6-v2',
                              batch_size: int = 64) -> np.ndarray:
    """Sentence embeddings через sentence-transformers (просто и быстро)"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # для cosine similarity
    )
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def get_bert_cls_embeddings(texts: list,
                             model_name: str = 'bert-base-multilingual-cased',
                             batch_size: int = 16,
                             device: str = None) -> np.ndarray:
    """CLS токен из BERT как embedding"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS токен
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)
```

---

## 5. Классификация текста (полный пайплайн)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def text_classification_pipeline(texts_train: list, y_train, texts_test: list,
                                   task: str = 'binary') -> dict:
    """Полный пайплайн классификации текста"""
    texts_train_proc = preprocess_text_pipeline(texts_train)
    texts_test_proc = preprocess_text_pipeline(texts_test)

    # Несколько моделей для сравнения
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=1000, C=5.0, random_state=42)),
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
        'Naive Bayes': Pipeline([
            ('tfidf', CountVectorizer(max_features=50000, ngram_range=(1, 2))),
            ('clf', MultinomialNB(alpha=0.1)),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, texts_train_proc, y_train, cv=5,
                                  scoring='f1_weighted', n_jobs=-1)
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
        pipe.fit(texts_train_proc, y_train)
        results[name] = {
            'cv_score': scores.mean(),
            'cv_std': scores.std(),
            'pipeline': pipe
        }

    # Лучшая модель
    best_name = max(results, key=lambda k: results[k]['cv_score'])
    best_pipe = results[best_name]['pipeline']
    y_pred = best_pipe.predict(texts_test_proc)

    return y_pred, results
```

---

## 6. Тематическое моделирование (LDA)

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def fit_lda_model(texts: list, n_topics: int = 10,
                   max_features: int = 5000, n_words_per_topic: int = 10):
    """Тематическое моделирование с LDA"""
    vectorizer = CountVectorizer(max_features=max_features, min_df=2, max_df=0.9)
    doc_term_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        random_state=42,
        n_jobs=-1,
        learning_method='online',
    )
    doc_topics = lda.fit_transform(doc_term_matrix)

    # Топ слова для каждой темы
    print("Topics:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vocab[i] for i in topic.argsort()[:-n_words_per_topic - 1:-1]]
        print(f"  Topic {topic_idx}: {', '.join(top_words)}")

    # Перплексия (меньше = лучше)
    print(f"\nPerplexity: {lda.perplexity(doc_term_matrix):.2f}")

    return lda, doc_topics, vectorizer
```
