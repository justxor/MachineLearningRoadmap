# ✅ Разбор Задачи 09 — NLP

**Задание:** [task-09-nlp.md](../practice-tasks/task-09-nlp.md)

---

## 🔑 Ключевые инсайты

1. **TF-IDF + LogReg** — сильный baseline для классификации текста. Быстро и интерпретируемо
2. **sublinear_tf=True** — log(1+tf) вместо tf. Снижает вес частых слов
3. **ngram_range=(1,2)** — биграммы часто улучшают качество (ловят "не хорошо")
4. **Char n-grams** — лучше для языков с опечатками и сленгом
5. **BERT >> TF-IDF** по качеству, но в 100x медленнее и требует GPU

---

## 💻 Полное решение

```python
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

# Данные из задачи
reviews_base = [
    "Отличный продукт, очень доволен покупкой! Рекомендую всем",
    "Ужасное качество, сломалось через неделю. Деньги на ветер",
    "Нормально, ничего особенного. Работает как заявлено",
    "Супер! Лучший продукт что я покупал за последние годы",
    "Очень разочарован. Не соответствует описанию вообще",
    "Хорошее соотношение цена/качество для своей категории",
    "Прислали бракованный товар, поддержка не отвечает",
    "Всё чётко, доставка быстрая, товар соответствует фото",
    "Среднее качество, есть лучше за те же деньги на рынке",
    "Полный хлам! Выбросил сразу. Требую возврат средств!!!",
]
reviews = reviews_base * 200
np.random.seed(42)
sentiments = np.random.choice(['positive', 'negative', 'neutral'], len(reviews), p=[0.5, 0.3, 0.2])
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments,
                   'rating': np.random.choice([1,2,3,4,5], len(reviews))})

# Часть 1: Предобработка
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation + '!?'))
    text = re.sub(r'\s+', ' ', text).strip()
    # Простые стоп-слова (замена nltk для быстроты)
    stop_words = {'и', 'в', 'не', 'на', 'что', 'это', 'как', 'за', 'из', 'но', 'я', 'по', 'а', 'от', 'то', 'же'}
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

df['review_clean'] = df['review'].apply(clean_text)
df['review_len'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()
df['exclamation_count'] = df['review'].str.count('!')
df['uppercase_ratio'] = df['review'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))

# Топ слова по классам
for sentiment in ['positive', 'negative']:
    texts = ' '.join(df[df['sentiment'] == sentiment]['review_clean'])
    words = Counter(texts.split()).most_common(15)
    print(f"\nТоп-15 слов ({sentiment}): {[w for w, _ in words]}")

# Часть 2: Векторизация
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['review_clean'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

vectorizers = {
    'CountVec': CountVectorizer(max_features=2000, ngram_range=(1,1)),
    'TF-IDF': TfidfVectorizer(max_features=2000, ngram_range=(1,2), sublinear_tf=True, min_df=2),
    'CharTFIDF': TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True),
}

classifiers = {
    'NaiveBayes': MultinomialNB(alpha=0.1),
    'LogisticRegression': LogisticRegression(C=5.0, max_iter=1000, random_state=42),
    'LinearSVC': LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results_table = {}
best_score = 0
best_combo = None
best_pipe = None

for vec_name, vec in vectorizers.items():
    for clf_name, clf in classifiers.items():
        combo = f"{vec_name} + {clf_name}"
        pipe = Pipeline([('vec', vec), ('clf', clf)])
        cv_scores = cross_val_score(pipe, X_train_raw, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        score = cv_scores.mean()
        results_table[combo] = {'F1-macro': score, 'Std': cv_scores.std()}
        if score > best_score:
            best_score = score
            best_combo = combo
            best_pipe = pipe

print(f"\nЛучшая комбинация: {best_combo} → F1-macro={best_score:.4f}")
print(pd.DataFrame(results_table).T.sort_values('F1-macro', ascending=False).round(4))

# Анализ ошибок
best_pipe.fit(X_train_raw, y_train)
y_pred = best_pipe.predict(X_test_raw)
print(classification_report(y_test, y_pred))

# Ошибки
errors_df = pd.DataFrame({'text': X_test_raw, 'true': y_test, 'pred': y_pred})
errors_df = errors_df[errors_df['true'] != errors_df['pred']]
print(f"\nОшибок: {len(errors_df)}")
print(errors_df.head(5)[['text', 'true', 'pred']].to_string())
# Типичные ошибки: нейтральный → позитивный (слова как "нормально", "работает")

# Топ слова LogReg
if 'LogisticRegression' in best_combo:
    vec_fitted = best_pipe.named_steps['vec']
    clf_fitted = best_pipe.named_steps['clf']
    if hasattr(clf_fitted, 'coef_'):
        classes = clf_fitted.classes_
        vocab = vec_fitted.get_feature_names_out()
        for i, cls in enumerate(classes):
            coefs = clf_fitted.coef_[i]
            top_words = [vocab[j] for j in coefs.argsort()[-10:][::-1]]
            print(f"Топ слова для '{cls}': {top_words}")
```

---

## 🐛 Типичные ошибки

1. **Не удалять стоп-слова** — "и", "в", "не" занимают топ TF-IDF без информации
2. **Не использовать sublinear_tf** — часто встречающееся слово не должно доминировать
3. **Слишком маленький max_features** — теряешь редкие но важные слова
4. **Не строить confusion matrix** — пары positive/negative и neutral/positive — разные ошибки

---

## 📌 Ключевые выводы

- **TF-IDF (1,2) + LogReg** — обычно лучшая комбинация для TF-IDF подходов
- **Neutral** класс сложнее всего — меньше явных сигналов
- **BERT в 2–3 раза лучше по F1** чем TF-IDF, но нужен GPU и больше данных
- **Char n-grams** хороши для языков с опечатками и морфологией (русский!)
