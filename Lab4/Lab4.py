#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №4. Линейные модели, SVM и деревья решений
# 
# **Курс:** Технологии машинного обучения
# 
# **Цель работы:** изучение линейных моделей, SVM и деревьев решений.
# 
# **Задание:**
# 1. Выбрать набор данных (датасет) для решения задачи классификации или регрессии.
# 2. В случае необходимости провести удаление или заполнение пропусков и кодирование категориальных признаков.
# 3. С использованием метода `train_test_split` разделить выборку на обучающую и тестовую.
# 4. Обучить модели: логистическую регрессию (линейная модель для классификации), SVM, дерево решений.
# 5. Оценить качество моделей с помощью двух подходящих для задачи метрик и сравнить модели.
# 6. Построить график важности признаков в дереве решений.
# 7. Визуализировать дерево решений и вывести его правила в текстовом виде.
# 
# Ссылка на постановку задания: https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_TREES

# ## 1. Выбор набора данных
# 
# Используется датасет **Palmer Archipelago (Antarctica) Penguins** с платформы Kaggle
# ([parulpandey/palmer-archipelago-antarctica-penguin-data](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)).
# Файл сохранён локально в `data/penguins.csv` (копия из открытого репозитория seaborn-data с теми же данными Palmer Station LTER).
# 
# Датасет содержит измерения 344 пингвинов трёх видов. Решается задача **многоклассовой классификации** —
# определить вид пингвина (`Adelie`, `Chinstrap`, `Gentoo`) по морфологическим признакам.
# 
# Датасет подходит для задания, так как:
# - есть **пропуски** (в числовых признаках и в `sex`), которые нужно обработать;
# - есть **категориальные признаки** (`island`, `sex`), которые нужно закодировать;
# - признаки разного масштаба (граммы и миллиметры), что важно для логистической регрессии и SVM;
# - небольшой размер и интерпретируемые признаки позволяют наглядно визуализировать дерево решений.
# 
# **Описание столбцов:**
# - `species` — вид пингвина (целевой признак);
# - `island` — остров наблюдения (Biscoe, Dream, Torgersen) — категориальный;
# - `bill_length_mm` — длина клюва (мм);
# - `bill_depth_mm` — глубина клюва (мм);
# - `flipper_length_mm` — длина плавника (мм);
# - `body_mass_g` — масса тела (г);
# - `sex` — пол (MALE / FEMALE) — категориальный.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42


# ## 2. Загрузка данных и первичный анализ

# In[2]:


data = pd.read_csv("data/penguins.csv")
print("Размер набора данных:", data.shape)
data.head()


# In[3]:


data.info()


# In[4]:


data.describe().T


# In[5]:


print("Распределение классов:")
print(data["species"].value_counts())

sns.pairplot(data, hue="species", vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
             palette="Set2", height=2.0, corner=True)
plt.show()


# ## 3. Обработка пропусков и кодирование категориальных признаков

# In[6]:


missing = pd.DataFrame({"Пропусков": data.isna().sum(),
                        "Доля, %": (data.isna().mean() * 100).round(1)})
missing


# Пропуски есть в четырёх числовых признаках (по 2 строки) и в `sex` (11 строк, ~3 %). Пропусков мало, но выбрасывать строки при
# выборке всего в 344 объекта нежелательно, поэтому:
# - в числовых признаках пропуски заполняются **медианой**;
# - в категориальных (`sex`) — **самым частым значением**;
# - категориальные признаки кодируются **One-Hot Encoding**;
# - числовые признаки масштабируются `StandardScaler` (нужно для логистической регрессии и SVM; для дерева не обязательно, но не вредит).
# 
# Медианы и частоты вычисляются только по обучающей выборке: вся предобработка собрана в `ColumnTransformer`, входящий в `Pipeline`, что исключает утечку данных.
# Строки, в которых пропущены все измерения (две записи), сохраняются, так как импьютер заполняет их по обучающей выборке.

# In[7]:


# на всякий случай проверим уникальные значения категориальных признаков
for col in ["island", "sex"]:
    print(col, "->", data[col].unique())


# ## 4. Разделение выборки на обучающую и тестовую

# In[8]:


X = data.drop(columns="species")
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

print("Обучающая выборка:", X_train.shape, " Тестовая выборка:", X_test.shape)
print("\nДоли классов в обучающей выборке:\n", y_train.value_counts(normalize=True).round(3))
print("\nДоли классов в тестовой выборке:\n", y_test.value_counts(normalize=True).round(3))


# In[9]:


num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_cols = ["island", "sex"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# Проверим результат предобработки на обучающей выборке
Xt = preprocess.fit_transform(X_train)
feature_names = list(preprocess.get_feature_names_out())
print("Признаки после предобработки:", feature_names)
print("Пропусков после обработки:", int(np.isnan(Xt).sum()))
pd.DataFrame(Xt, columns=feature_names).head()


# ## 5. Обучение моделей
# 
# Обучаются три модели (каждая в составе `Pipeline` с общей предобработкой):
# 1. **Логистическая регрессия** (`LogisticRegression`) — линейная модель классификации;
# 2. **SVM** (`SVC` с RBF-ядром);
# 3. **Дерево решений** (`DecisionTreeClassifier`, `max_depth=4` — ограничение глубины для читаемости и защиты от переобучения).

# In[10]:


models = {
    "Логистическая регрессия": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
    "Дерево решений": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
}

fitted = {}
for name, est in models.items():
    fitted[name] = Pipeline([("prep", preprocess), ("model", est)]).fit(X_train, y_train)
    print("Обучена модель:", name)


# ## 6. Оценка качества и сравнение моделей
# 
# Задача многоклассовая, классы умеренно несбалансированы (Adelie ~44 %, Gentoo ~36 %, Chinstrap ~20 %), поэтому используются две метрики:
# - **Accuracy** — доля верных ответов;
# - **F1-macro** — среднее F1 по классам (одинаково учитывает редкий класс Chinstrap).
# 
# Дополнительно приведены macro-precision и macro-recall, а также 5-кратная кросс-валидация на обучающей выборке для оценки устойчивости.

# In[11]:


rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, model in fitted.items():
    pred = model.predict(X_test)
    cv_acc = cross_val_score(Pipeline([("prep", preprocess), ("model", models[name])]),
                             X_train, y_train, cv=cv, scoring="accuracy")
    rows.append({
        "Модель": name,
        "Accuracy": accuracy_score(y_test, pred),
        "F1-macro": f1_score(y_test, pred, average="macro"),
        "Precision-macro": precision_score(y_test, pred, average="macro"),
        "Recall-macro": recall_score(y_test, pred, average="macro"),
        "CV Accuracy (train, 5 fold)": cv_acc.mean(),
    })
results = pd.DataFrame(rows).set_index("Модель")
results.round(4)


# In[12]:


ax = results[["Accuracy", "F1-macro"]].plot(kind="bar", figsize=(8, 4), rot=0, colormap="Set2", width=0.7)
ax.set_ylim(0.8, 1.02)
ax.set_title("Сравнение моделей на тестовой выборке")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=9, padding=2)
plt.tight_layout()
plt.show()


# In[13]:


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
    ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()


# In[14]:


for name, model in fitted.items():
    print("=" * 60)
    print(name)
    print(classification_report(y_test, model.predict(X_test), digits=3))


# ## 7. Важность признаков в дереве решений

# In[15]:


tree_model = fitted["Дерево решений"].named_steps["model"]
importances = pd.Series(tree_model.feature_importances_, index=feature_names).sort_values()
importances.index = [n.split("__", 1)[1] for n in importances.index]

print(importances.sort_values(ascending=False).round(4))

fig, ax = plt.subplots(figsize=(8, 4.5))
importances.plot(kind="barh", color="teal", ax=ax)
ax.set_title("Важность признаков в дереве решений")
ax.set_xlabel("Важность (снижение неопределённости Gini)")
plt.tight_layout()
plt.show()


# ## 8. Визуализация дерева решений и правила в текстовом виде

# In[16]:


short_names = [n.split("__", 1)[1] for n in feature_names]

fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(tree_model, feature_names=short_names, class_names=list(tree_model.classes_),
          filled=True, rounded=True, fontsize=9, impurity=False, ax=ax)
ax.set_title("Дерево решений (max_depth=4). Признаки стандартизованы")
plt.tight_layout()
plt.show()


# In[17]:


print(export_text(tree_model, feature_names=short_names))


# Признаки в дереве стандартизованы (среднее 0, стандартное отклонение 1), поэтому пороги в правилах выражены в единицах стандартного отклонения.
# При необходимости их можно перевести в исходные единицы: `x = mean + z · std`.

# In[18]:


# Перевод порогов первых разбиений в исходные единицы измерения
scaler = fitted["Дерево решений"].named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
stats = pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=num_cols)

t = tree_model.tree_
rows = []
for node in range(t.node_count):
    if t.children_left[node] != -1:
        fname = short_names[t.feature[node]]
        thr = t.threshold[node]
        if fname in stats.index:
            rows.append({"Узел": node, "Признак": fname, "Порог (z)": round(thr, 3),
                         "Порог (исходные ед.)": round(stats.loc[fname, "mean"] + thr * stats.loc[fname, "std"], 2)})
        else:
            rows.append({"Узел": node, "Признак": fname, "Порог (z)": round(thr, 3), "Порог (исходные ед.)": "категория (0/1)"})
pd.DataFrame(rows)


# ## 9. Выводы
# 
# - Выбран датасет Palmer Penguins (Kaggle): многоклассовая классификация вида пингвина. Пропуски (числовые — медиана, `sex` — мода) заполнены, категориальные признаки `island` и `sex` закодированы One-Hot, числовые стандартизованы; вся предобработка внутри `Pipeline`, утечки данных нет.
# - Выборка разделена `train_test_split` (75/25, стратификация по виду).
# - Обучены логистическая регрессия, SVM (RBF) и дерево решений (`max_depth=4`). Качество оценено по Accuracy и F1-macro.
# - На тестовой выборке логистическая регрессия и SVM дали 100 % по обеим метрикам, дерево решений — Accuracy 0.988 и F1-macro 0.990 (1 ошибка из 86). По 5-кратной кросс-валидации на обучающей выборке модели ранжируются так же: логистическая регрессия (0.992) > SVM (0.988) > дерево (0.973).
#   Тестовая выборка мала (86 объектов), поэтому различие в один объект несущественно: все три модели решают задачу практически идеально, так как виды хорошо разделяются по размерам плавника и клюва, то есть классы почти линейно разделимы.
# - Наиболее важные признаки в дереве: `flipper_length_mm` (0.52) и `bill_length_mm` (0.34), затем `island_Dream` (0.10). Остальные признаки почти не используются. Это согласуется с парными графиками: Gentoo отделяется по длине плавника, а Adelie и Chinstrap — по длине клюва.
# - Правила дерева (`export_text`) интерпретируемы: например, если длина плавника > 206.5 мм и остров не Dream, то вид — Gentoo.
