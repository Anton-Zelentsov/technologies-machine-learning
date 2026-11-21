# -*- coding: utf-8 -*-
"""
Рубежный контроль №1 по курсу «Технологии разведочного анализа и обработки данных».

Вариант №9: задача №2, набор данных №1 — Iris (sklearn.datasets.load_iris).

Задание: провести обработку пропусков в данных для одного категориального
и одного количественного признака; ответить, какие способы обработки пропусков
использованы и какие признаки будут использоваться для построения моделей.

Скрипт соответствует ноутбуку РК_1.ipynb и запускается из папки проекта:
    python РК_1.py

Результаты: data/iris.csv, data/iris_missing.csv, data/iris_prepared.csv,
графики в папке figures/.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Кириллица в подписях графиков
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="DejaVu Sans")

RANDOM_STATE = 42
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name):
    """Сохраняет рисунок в папку figures — эти же файлы вставлены в отчет."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")


print("pandas:", pd.__version__, "| numpy:", np.__version__)
print("scikit-learn:", __import__("sklearn").__version__, "| seaborn:", sns.__version__)


iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# В Iris нет категориальных признаков, поэтому создаем его из количественного sepal_width:
# разбиваем значения на три равные по числу наблюдений группы (terciles)
df["sepal_width_cat"] = pd.qcut(df["sepal_width"], q=3,
                                labels=["узкая", "средняя", "широкая"]).astype(object)

# В Iris нет пропусков, поэтому вносим их искусственно (примечание к заданию)
rng = np.random.default_rng(RANDOM_STATE)
idx_cat = rng.choice(df.index, size=18, replace=False)                      # 12 % категориального
idx_num = rng.choice(df.index.difference(idx_cat), size=15, replace=False)  # 10 % количественного

df_clean = df.copy()          # эталон без пропусков: истинные значения известны
df_miss = df.copy()           # набор с пропусками
df_miss.loc[idx_cat, "sepal_width_cat"] = np.nan
df_miss.loc[idx_num, "petal_length"] = np.nan

df.to_csv(DATA_DIR / "iris.csv", index=False)
df_miss.to_csv(DATA_DIR / "iris_missing.csv", index=False)

num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

print("Размер набора:", df_miss.shape)
print("\nПропуски по столбцам:")
print(df_miss.isna().sum()[df_miss.isna().sum() > 0].to_string())
print("\nДоля пропусков:")
print((df_miss.isna().mean()[df_miss.isna().mean() > 0] * 100).round(1).to_string(), "%")
print("\nЧисло объектов по видам:", df_miss["species"].value_counts().sort_index().to_dict())
print(df_miss.head())
print(df_miss[num_cols].describe().round(3))


fig, ax = plt.subplots(figsize=(11, 3))
sns.heatmap(df_miss[["petal_length", "sepal_width_cat"]].isna().T, cbar=False,
            cmap="Greys", ax=ax)
ax.set_title("Карта пропусков: темное — пропуск")
ax.set_xlabel("Номер объекта")
ax.set_yticklabels(["petal_length (10 %)", "sepal_width_cat (12 %)"], rotation=0)
plt.tight_layout()
save_fig(fig, "fig01_missing_map.png")
plt.show()


# Проверяем, есть ли корреляции, на которых можно построить модельное заполнение
corr = df_clean[num_cols].corr()
print("Корреляция Пирсона (по полному набору без пропусков):")
print(corr.round(3))
print("Связь petal_length с остальными признаками:")
print(corr["petal_length"].drop("petal_length").round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            square=True, ax=ax)
ax.set_title("Корреляция признаков Iris")
plt.tight_layout()
save_fig(fig, "fig02_corr.png")
plt.show()


# ---------- Количественный признак petal_length: 15 пропусков (10 %) ----------
true_num = df_clean.loc[idx_num, "petal_length"]   # истинные значения (пропуски внесены нами)

# Способ 1: удаление строк
rows_dropped = df_miss.dropna(subset=["petal_length"])
print(f"Удаление строк: осталось {len(rows_dropped)} из {len(df_miss)} наблюдений "
      f"(потеряно {len(df_miss) - len(rows_dropped)} строк, "
      f"{100 * (1 - len(rows_dropped) / len(df_miss)):.1f} %)")

# Способы 2-4: заполнение средним, медианой и методом k ближайших соседей
filled = {
    "Среднее": pd.Series(
        SimpleImputer(strategy="mean").fit_transform(df_miss[["petal_length"]])[:, 0],
        index=df_miss.index),
    "Медиана": pd.Series(
        SimpleImputer(strategy="median").fit_transform(df_miss[["petal_length"]])[:, 0],
        index=df_miss.index),
    "KNN (k=5)": pd.Series(
        KNNImputer(n_neighbors=5).fit_transform(df_miss[num_cols])[:, num_cols.index("petal_length")],
        index=df_miss.index),
}

rows = []
for name, series in filled.items():
    pred = series.loc[idx_num]
    rows.append({"Способ заполнения": name,
                 "MAE, см": mean_absolute_error(true_num, pred),
                 "RMSE, см": np.sqrt(mean_squared_error(true_num, pred))})
impute_metrics = pd.DataFrame(rows).set_index("Способ заполнения").round(3)
print("\nКачество восстановления petal_length (сравнение с истинными значениями):")
print(impute_metrics)
print("Лучший способ по MAE:", impute_metrics["MAE, см"].idxmin())


fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
sns.histplot(df_miss["petal_length"].dropna(), bins=15, kde=True, ax=axes[0], color="#8c8c8c")
axes[0].set_title(f"До обработки\n(пропусков: {int(df_miss['petal_length'].isna().sum())})")
for ax, name, color in zip(axes[1:], ["Среднее", "Медиана", "KNN (k=5)"],
                           ["#DD8452", "#8172B2", "#4C72B0"]):
    sns.histplot(filled[name], bins=15, kde=True, ax=ax, color=color)
    ax.set_title(f"Заполнение: {name}")
for ax in axes:
    ax.set_xlabel("petal_length, см")
plt.tight_layout()
save_fig(fig, "fig03_num_impute_hist.png")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, metric, color in zip(axes, ["MAE, см", "RMSE, см"], ["#4C72B0", "#DD8452"]):
    impute_metrics[metric].plot(kind="bar", ax=ax, color=color, rot=15)
    ax.set_title(f"{metric.split(',')[0]} восстановления petal_length")
    ax.set_ylabel(metric)
    for patch in ax.patches:
        ax.annotate(f"{patch.get_height():.3f}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
save_fig(fig, "fig04_num_impute_metrics.png")
plt.show()


# ---------- Категориальный признак sepal_width_cat: 18 пропусков (12 %) ----------
mode_value = df_miss["sepal_width_cat"].mode()[0]

# Способ 1: заполнение модой
cat_mode = df_miss["sepal_width_cat"].fillna(mode_value)
# Способ 2: замена отдельной категорией "неизвестно"
cat_unknown = df_miss["sepal_width_cat"].fillna("неизвестно")
# Для сравнения: удаление строк
cat_dropped = df_miss["sepal_width_cat"].dropna()

true_cat = df_clean.loc[idx_cat, "sepal_width_cat"]
mode_accuracy = (cat_mode.loc[idx_cat].astype(str).values == true_cat.astype(str).values).mean()
print(f"Мода категориального признака: '{mode_value}' ({int((df_miss['sepal_width_cat'] == mode_value).sum())} наблюдений)")
print(f"Доля верно восстановленных категорий способом 'мода': {mode_accuracy:.1%}")
print(f"Удаление строк: осталось {len(cat_dropped)} из {len(df_miss)} наблюдений")

cat_compare = pd.DataFrame({
    "Исходные (без пропусков)": df_miss["sepal_width_cat"].value_counts(normalize=True),
    "Заполнение модой": cat_mode.value_counts(normalize=True),
    "Категория 'неизвестно'": cat_unknown.value_counts(normalize=True),
}).fillna(0).round(3)
print("\nРаспределение категорий (доли):")
print(cat_compare)


fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
order = ["узкая", "средняя", "широкая", "неизвестно"]
for ax, title, series in zip(
        axes,
        ["До обработки", "Заполнение модой", "Категория 'неизвестно'"],
        [df_miss["sepal_width_cat"], cat_mode, cat_unknown]):
    present = [o for o in order if o in set(series)]
    sns.countplot(x=series, order=present, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("sepal_width_cat")
    ax.set_ylabel("Число наблюдений")
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height())}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
save_fig(fig, "fig05_cat_impute.png")
plt.show()

fig, ax = plt.subplots(figsize=(9, 3.4))
cat_compare.T.plot(kind="bar", rot=0, ax=ax, colormap="Set2", width=0.8)
ax.set_title("Распределение категорий при разных способах обработки")
ax.set_ylabel("Доля наблюдений")
ax.set_xlabel("Способ обработки")
ax.legend(title="Категория", ncol=4, fontsize=8)
plt.tight_layout()
save_fig(fig, "fig06_cat_compare.png")
plt.show()


# ---------- Итоговый набор: KNN для количественного, мода для категориального ----------
df_prepared = df_miss.copy()
df_prepared["petal_length"] = filled["KNN (k=5)"]
df_prepared["sepal_width_cat"] = cat_mode

print("Размер итогового набора:", df_prepared.shape)
print("Пропусков после обработки:", int(df_prepared.isna().sum().sum()))

feature_cols = num_cols + ["sepal_width_cat"]
final_df = df_prepared[feature_cols + ["species"]].copy()
final_df.to_csv(DATA_DIR / "iris_prepared.csv", index=False)

print("\nТипы данных:")
print(final_df.dtypes.to_string())
print(final_df.head())

# Проверяем, что заполнение не исказило распределения
check = pd.DataFrame({
    "Исходный (без пропусков)": df_clean[num_cols].agg(["mean", "std", "median"]).stack(),
    "После обработки": df_prepared[num_cols].agg(["mean", "std", "median"]).stack(),
}).unstack().round(3)
print("\nСравнение статистик до и после обработки пропусков:")
print(check)


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=pd.concat([df_clean[["petal_length"]].assign(Набор="Исходный"),
                            df_prepared[["petal_length"]].assign(Набор="После обработки")]),
            x="Набор", y="petal_length", ax=axes[0], palette="Set2")
axes[0].set_title("petal_length: исходный набор и после обработки пропусков")
sns.violinplot(data=final_df, x="species", y="petal_length", palette="Set3", ax=axes[1], cut=0)
axes[1].set_title("petal_length по видам после обработки пропусков")
plt.tight_layout()
save_fig(fig, "fig07_final_check.png")
plt.show()

pair = sns.pairplot(final_df, vars=num_cols, hue="species", palette="Set1",
                    diag_kind="hist", plot_kws={"s": 30, "alpha": 0.8})
pair.figure.suptitle("Итоговый набор после обработки пропусков", y=1.01)
save_fig(pair.figure, "fig08_final_pairplot.png")
plt.show()


# ---------- Обоснование выбора признаков для модели ----------
def eta_squared(values, groups):
    """Корреляционное отношение eta: сила связи количественного признака с классом."""
    values, groups = pd.Series(values), pd.Series(groups)
    grand_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in values.groupby(groups))
    ss_total = ((values - grand_mean) ** 2).sum()
    return ss_between / ss_total


eta = pd.Series({c: eta_squared(final_df[c], final_df["species"]) for c in num_cols}).sort_values(ascending=False)
print("Корреляционное отношение eta (связь признака с видом):")
print(eta.round(3).to_frame("eta"))

fig, ax = plt.subplots(figsize=(7, 3.4))
eta.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Вклад признаков в разделение видов (eta)")
ax.set_xlabel("eta")
for i, v in enumerate(eta.sort_values()):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05)
plt.tight_layout()
save_fig(fig, "fig09_eta.png")
plt.show()

print("\nМультиколлинеарность (|r| > 0,8):")
for i, a in enumerate(num_cols):
    for b in num_cols[i + 1:]:
        if abs(corr.loc[a, b]) > 0.8:
            print(f"  {a} — {b}: r = {corr.loc[a, b]:.3f}")
