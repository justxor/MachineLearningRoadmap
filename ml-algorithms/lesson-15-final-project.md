# Урок 15: Итоговый проект — ML Pipeline от А до Я

**Курс:** [Алгоритмы Машинного Обучения](README.md) | [← Урок 14](lesson-14-tsne-umap.md)

---

## 🎯 Цель проекта

Применить **все знания курса** в полном ML пайплайне на реальных данных. Проект подходит как портфельная работа.

**Датасет:** [Titanic](https://www.kaggle.com/c/titanic) или [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 📋 Структура проекта

```
project/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensemble.ipynb
├── src/
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── submission.csv
└── README.md
```

---

## 💻 Полный ML Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ШАГ 1: ЗАГРУЗКА И EDA
# ============================================================

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=== Train shape:", train.shape)
print("=== Test shape:", test.shape)
print("\n=== Target distribution:")
print(train['Survived'].value_counts(normalize=True))

print("\n=== Missing values (train):")
print(train.isnull().sum()[train.isnull().sum() > 0])

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Целевая переменная
axes[0, 0].pie(train['Survived'].value_counts(), labels=['Died', 'Survived'],
               autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Survival Rate')

# Выживаемость по полу
train.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Survival Rate by Sex')
axes[0, 1].tick_params(rotation=0)

# Выживаемость по классу
train.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Survival Rate by Class')
axes[0, 2].tick_params(rotation=0)

# Распределение возраста
axes[1, 0].hist(train['Age'].dropna(), bins=30, edgecolor='black')
axes[1, 0].set_title('Age Distribution')

# Корреляционная матрица
numeric_cols = train.select_dtypes(include=[np.number]).columns
corr = train[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

# Pairplot для ключевых признаков
train[['Age', 'Fare', 'Pclass', 'Survived']].dropna().plot.scatter(
    x='Age', y='Fare', c='Survived', colormap='viridis', ax=axes[1, 2])
axes[1, 2].set_title('Age vs Fare (colored by Survived)')

plt.tight_layout(); plt.savefig('eda.png', dpi=150); plt.show()


# ============================================================
# ШАГ 2: FEATURE ENGINEERING
# ============================================================

def create_features(df):
    df = df.copy()
    
    # 1. Заполнение пропусков
    df['Age'] = df['Age'].fillna(df.groupby(['Sex', 'Pclass'])['Age'].transform('median'))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # 2. Новые признаки
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                        'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    df['HasCabin'] = (~df['Cabin'].isna()).astype(int)
    
    # 3. Кодирование категориальных признаков
    cat_cols = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'Deck']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 4. Отбор признаков
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'Title', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin', 'HasCabin', 'Deck']
    
    return df[feature_cols]


X = create_features(train)
y = train['Survived']
X_test_final = create_features(test)

print("\nFeature shape:", X.shape)
print("Features:", X.columns.tolist())


# ============================================================
# ШАГ 3: BASELINE И СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'GBM': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGB': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0),
    'LGB': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                random_state=42, n_jobs=-1, verbose=-1)
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = scores
    print(f"{name:5s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Визуализация сравнения
fig, ax = plt.subplots(figsize=(10, 5))
positions = range(len(results))
means = [v.mean() for v in results.values()]
stds = [v.std() for v in results.values()]
ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.8, color='steelblue')
ax.set_xticks(positions); ax.set_xticklabels(list(results.keys()))
ax.set_ylabel('CV Accuracy'); ax.set_title('Model Comparison')
ax.set_ylim(0.75, 0.90)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.002, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.show()


# ============================================================
# ШАГ 4: ПОДБОР ГИПЕРПАРАМЕТРОВ ЛУЧШЕЙ МОДЕЛИ
# ============================================================

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0
    }
    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgb, n_trials=50)
print(f"\nBest XGB accuracy: {study.best_value:.4f}")

# Обучение лучшей модели
best_xgb = xgb.XGBClassifier(**{k: v for k, v in study.best_params.items() 
                                  if k != 'lr'}, learning_rate=study.best_params['lr'],
                               random_state=42, n_jobs=-1, verbosity=0)
best_xgb.fit(X_scaled, y)


# ============================================================
# ШАГ 5: СТЕКИНГ-АНСАМБЛЬ
# ============================================================

def get_oof_predictions(model, X, y, X_test, cv, scaler=None):
    """Out-of-Fold предсказания."""
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
    
    return oof_preds, test_preds

X_test_scaled = scaler.transform(X_test_final)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ('xgb', best_xgb),
    ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1))
]

train_oof = np.zeros((len(y), len(base_models)))
test_oof = np.zeros((len(X_test_scaled), len(base_models)))

print("\n=== Stacking: OOF Predictions ===")
for i, (name, model) in enumerate(base_models):
    oof, tst = get_oof_predictions(model, X_scaled, y, X_test_scaled, cv)
    train_oof[:, i] = oof
    test_oof[:, i] = tst
    auc = roc_auc_score(y, oof)
    print(f"{name}: OOF AUC = {auc:.4f}")

# Мета-модель
meta_model = LogisticRegression(C=0.1, max_iter=1000)
meta_model.fit(train_oof, y)
final_preds = meta_model.predict(test_oof)
final_probs = meta_model.predict_proba(train_oof)[:, 1]
print(f"\nStacking OOF AUC: {roc_auc_score(y, final_probs):.4f}")


# ============================================================
# ШАГ 6: ФИНАЛЬНЫЙ SUBMISSION
# ============================================================

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_preds
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved! Preview:")
print(submission.head(10))
print(f"Survival rate: {submission['Survived'].mean():.3f}")


# ============================================================
# ШАГ 7: ИНТЕРПРЕТАЦИЯ МОДЕЛИ
# ============================================================

# Feature Importance
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(feature_importance)), feature_importance['importance'])
ax.set_xticks(range(len(feature_importance)))
ax.set_xticklabels(feature_importance['feature'], rotation=45, ha='right')
ax.set_title('Feature Importance (Random Forest)')
ax.set_ylabel('Importance')
plt.tight_layout(); plt.show()

print("\n=== Top Features ===")
print(feature_importance.head(8).to_string(index=False))
```

---

## 📊 Чеклист проекта

- [ ] EDA: анализ распределений, пропусков, корреляций
- [ ] Feature Engineering: создано не менее 5 новых признаков
- [ ] Baseline: LogReg / простое дерево как отправная точка
- [ ] Сравнение 4+ моделей через 5-fold CV
- [ ] Подбор гиперпараметров лучшей модели (Optuna, 30+ trials)
- [ ] Stacking ансамбль из 3 разнородных моделей
- [ ] Финальный submission
- [ ] Интерпретация: feature importance, примеры предсказаний
- [ ] README с описанием подхода и результатами

---

## 🎯 Ожидаемый результат

| Этап | Accuracy (Titanic) |
|------|-------------------|
| Baseline (LogReg) | ~0.79 |
| Random Forest | ~0.82 |
| XGBoost tuned | ~0.83 |
| Stacking | ~0.84+ |

---

## 🔗 Что дальше после курса

- **[Нейронные сети](../neural-networks/README.md)** — Deep Learning с PyTorch
- **[Data Science](../data-science/README.md)** — продакшн ML системы  
- **[Math for ML](../math-for-ml/README.md)** — углублённая математика
- **[LLM Engineering](../llm-engineering/README.md)** — RAG, Fine-tuning, AI Agents

---

🎉 **Поздравляем с завершением курса "Алгоритмы Машинного Обучения"!**

Вы изучили 15 алгоритмов от линейной регрессии до UMAP. Теперь у вас есть твёрдый фундамент для Deep Learning и LLM Engineering.

**[← Назад к курсу](README.md)**
