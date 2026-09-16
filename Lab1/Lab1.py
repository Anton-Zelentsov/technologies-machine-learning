#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №1. Разведочный анализ данных (EDA) и визуализация
# 
# **Датасет:** Wine recognition dataset (Scikit-learn)
# 
# **Курс:** Технологии машинного обучения
# 
# **Задание:** выбрать датасет без пропусков, описать его, изучить основные характеристики, провести визуальное исследование и проанализировать корреляции признаков.
# 

# ## 1. Текстовое описание набора данных
# 
# Используется встроенный в Scikit-learn датасет **Wine recognition dataset** (`sklearn.datasets.load_wine`).
# 
# Это результаты химического анализа вин, выращенных в одном регионе Италии, но полученных из трёх разных сортов винограда (культиваров). Для каждого образца вина измерено 13 количественных характеристик, полученных методом химического анализа (содержание алкоголя, яблочной кислоты, золы, магния, фенолов и т.д.).
# 
# **Задача**, для которой обычно используется датасет — классификация: по химическому составу определить, к какому из трёх сортов винограда (`class_0`, `class_1`, `class_2`) относится вино.
# 
# Датасет выбран, поскольку:
# - не содержит пропущенных значений (что рекомендовано для первой лабораторной работы);
# - все признаки количественные, что удобно для изучения корреляций;
# - имеет разумный размер (178 наблюдений × 13 признаков) — не перегружает визуализацию;
# - есть целевая переменная (класс вина), что позволяет исследовать различия между группами.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["target_name"] = df["target"].map(dict(enumerate(wine.target_names)))

df.head()


# ## 2. Основные характеристики датасета
# 
# Рассмотрим размер таблицы, типы признаков, наличие пропусков и базовую описательную статистику.
# 

# In[2]:


print(f"Размер датасета: {df.shape[0]} наблюдений, {df.shape[1]} столбцов")
print()
print("Информация о столбцах:")
df.info()


# In[3]:


# Проверка пропущенных значений
missing = df.isna().sum()
print("Пропущенные значения по столбцам:")
print(missing)
print()
print(f"Всего пропусков в датасете: {missing.sum()}")


# In[4]:


# Распределение по классам (сортам вина)
class_counts = df["target_name"].value_counts()
print("Распределение наблюдений по классам:")
print(class_counts)


# In[5]:


# Описательная статистика по числовым признакам
df.drop(columns=["target"]).describe().T


# **Краткие выводы по характеристикам:**
# - Датасет полностью без пропусков (0 пропущенных значений во всех 14 столбцах).
# - 178 наблюдений, 13 количественных признаков + целевая переменная `target` (3 класса).
# - Классы не идеально сбалансированы: `class_1` — 71 образец, `class_0` — 59, `class_2` — 48.
# - Признаки имеют очень разный масштаб (например, `proline` — сотни единиц, `hue` — доли единицы), что важно учитывать при дальнейшем моделировании (потребуется масштабирование).
# 

# ## 3. Визуальное исследование датасета
# 
# ### 3.1. Распределение классов
# 

# In[6]:


fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="target_name", hue="target_name", palette="viridis", legend=False, ax=ax)
ax.set_title("Количество наблюдений по сортам вина")
ax.set_xlabel("Сорт вина")
ax.set_ylabel("Количество наблюдений")
plt.tight_layout()
plt.show()


# ### 3.2. Распределения отдельных признаков (гистограммы)

# In[7]:


features = wine.feature_names
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(data=df, x=feature, kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(feature, fontsize=10)
    axes[i].set_xlabel("")

for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Распределения признаков", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# ### 3.3. Boxplot ключевых признаков по классам
# 
# Посмотрим, как различаются классы вин по нескольким наиболее интерпретируемым признакам.

# In[8]:


key_features = ["alcohol", "flavanoids", "color_intensity", "proline"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, feature in zip(axes, key_features):
    sns.boxplot(data=df, x="target_name", y=feature, hue="target_name", palette="Set2", legend=False, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("Сорт вина")

plt.tight_layout()
plt.show()


# По этим графикам уже видно, что признаки `flavanoids`, `color_intensity` и `proline` заметно различаются между классами — они потенциально наиболее информативны для классификации сортов вина.

# ### 3.4. Попарные диаграммы рассеяния (pairplot) для ключевых признаков

# In[9]:


subset_features = ["alcohol", "flavanoids", "color_intensity", "proline", "target_name"]
sns.pairplot(df[subset_features], hue="target_name", palette="Set2", diag_kind="kde", corner=True)
plt.suptitle("Попарные зависимости ключевых признаков", y=1.02)
plt.show()


# Видно, что классы довольно неплохо разделяются уже по паре признаков (`flavanoids` и `proline` в частности), что подтверждает потенциальную пригодность датасета для задачи классификации.

# ## 4. Информация о корреляции признаков
# 
# Так как все признаки количественные, посчитаем матрицу корреляции Пирсона и визуализируем её тепловой картой.
# 

# In[10]:


corr = df[wine.feature_names].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Матрица корреляции признаков (Pearson)")
plt.tight_layout()
plt.show()


# In[11]:


# Топ-10 самых сильных попарных корреляций (по модулю), исключая диагональ
corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
top_corr = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).head(10)
top_corr.to_frame("correlation")


# **Наблюдения по корреляциям:**
# 
# - Наиболее сильная положительная связь — между `flavanoids` и `total_phenols` (это ожидаемо: флавоноиды — основная группа фенольных соединений в вине).
# - `flavanoids` также сильно коррелирует с `od280/od315_of_diluted_wines` (показатель, связанный с оптической плотностью, часто используется как косвенная мера концентрации фенолов).
# - `hue` имеет заметную отрицательную корреляцию с `color_intensity` — чем интенсивнее цвет, тем ниже оттеночный показатель.
# - Признак `proline` слабо коррелирует с большинством остальных признаков, что делает его независимым источником информации — это согласуется с его высокой различающей способностью между классами, увиденной на boxplot.
# - Признаки `alcohol` и `color_intensity` также умеренно положительно связаны.
# 
# Такая структура корреляций полезна для дальнейшего отбора признаков: сильно скоррелированные пары (`flavanoids`/`total_phenols`, `flavanoids`/`od280_od315`) несут во многом дублирующую информацию, тогда как `proline` и `alcohol` добавляют независимую информацию.
# 

# ## Итоговые выводы
# 
# 1. Датасет Wine recognition из Scikit-learn содержит 178 наблюдений, 13 количественных признаков без пропусков и целевую переменную с 3 классами.
# 2. Классы сортов вина не идеально сбалансированы (48–71 наблюдение на класс), что стоит учитывать при построении моделей классификации.
# 3. Признаки имеют разные масштабы измерения — для моделирования потребуется стандартизация/нормализация.
# 4. Визуальный анализ (гистограммы, boxplot, pairplot) показал, что признаки `flavanoids`, `color_intensity`, `proline` хорошо разделяют классы вин.
# 5. Анализ корреляций выявил сильные связи между фенольными показателями (`flavanoids`, `total_phenols`, `od280/od315_of_diluted_wines`) и относительную независимость признака `proline`, который при этом является одним из наиболее информативных для различения классов.
# 
