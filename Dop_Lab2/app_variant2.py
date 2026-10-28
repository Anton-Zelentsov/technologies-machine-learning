"""Вариант 2. Веб-приложение для нескольких моделей машинного обучения.

Запуск: streamlit run app_variant2.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import load_data, split_data, evaluate, plot_confusion, plot_roc

st.set_page_config(page_title="Вариант 2: сравнение моделей", layout="wide")
st.title("Вариант 2. Сравнение нескольких моделей: диагностика опухоли")
st.caption("Датасет: Breast Cancer Wisconsin (Kaggle). Выберите модели, задайте ключевой гиперпараметр и обучите их на одной выборке.")

df = load_data()

# модель -> (название гиперпараметра, (мин, макс, по умолчанию), фабрика модели)
MODELS = {
    "Логистическая регрессия": ("C", (0.01, 10.0, 1.0),
                                lambda v: make_pipeline(StandardScaler(), LogisticRegression(C=v, max_iter=1000))),
    "k ближайших соседей": ("n_neighbors", (1, 30, 5),
                            lambda v: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(v)))),
    "SVM (RBF)": ("C", (0.01, 10.0, 1.0),
                  lambda v: make_pipeline(StandardScaler(), SVC(C=v, probability=True, random_state=42))),
    "Дерево решений": ("max_depth", (1, 15, 4),
                       lambda v: DecisionTreeClassifier(max_depth=int(v), random_state=42)),
    "Случайный лес": ("n_estimators", (10, 300, 100),
                      lambda v: RandomForestClassifier(n_estimators=int(v), random_state=42)),
    "Градиентный бустинг": ("n_estimators", (10, 300, 100),
                            lambda v: GradientBoostingClassifier(n_estimators=int(v), random_state=42)),
}

# ---- выбор моделей и гиперпараметров ----
st.sidebar.header("Модели для обучения")
selected = st.sidebar.multiselect("Выберите модели", list(MODELS), default=["Логистическая регрессия", "Случайный лес"])
params = {}
for name in selected:
    hp, (lo, hi, default), _ = MODELS[name]
    params[name] = st.sidebar.slider(f"{name}: {hp}", lo, hi, default)
st.sidebar.header("Разбиение выборки")
test_size = st.sidebar.slider("Доля тестовой выборки", 0.1, 0.5, 0.25, step=0.05)
seed = st.sidebar.number_input("random_state", 0, 9999, 42)

with st.expander("Данные", expanded=False):
    st.write(f"Размер: {df.shape[0]} объектов, {df.shape[1] - 1} признаков.")
    st.dataframe(df.head(10))

# ---- обучение ----
if st.sidebar.button("Обучить выбранные модели", type="primary"):
    if not selected:
        st.sidebar.warning("Выберите хотя бы одну модель.")
    else:
        X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
        out = {}
        for name in selected:
            model = MODELS[name][2](params[name]).fit(X_train, y_train)
            metrics, pred, proba = evaluate(model, X_test, y_test)
            out[name] = {"metrics": metrics, "pred": pred, "proba": proba,
                         "hp": f"{MODELS[name][0]} = {params[name]}"}
        st.session_state["results"] = {"models": out, "y_test": y_test}

# ---- просмотр результатов ----
res = st.session_state.get("results")
if res is None:
    st.info("Выберите модели в боковой панели и нажмите «Обучить выбранные модели».")
else:
    models, y_test = res["models"], res["y_test"]
    table = pd.DataFrame({n: m["metrics"] for n, m in models.items()}).T
    table.insert(0, "Гиперпараметр", [m["hp"] for m in models.values()])

    st.subheader("Сравнение моделей на тестовой выборке")
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns[1:]})
                 .highlight_max(subset=table.columns[1:], color="#c8e6c9"), width="stretch")
    best = table["ROC AUC"].idxmax()
    st.success(f"Лучшая модель по ROC AUC: {best} ({table.loc[best, 'ROC AUC']:.3f})")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        table.drop(columns="Гиперпараметр").plot.bar(ax=ax, width=0.8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylim(0.7, 1.0)
        ax.set_title("Метрики моделей")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
    with c2:
        st.pyplot(plot_roc(y_test, {n: m["proba"] for n, m in models.items()}))

    st.subheader("Матрицы ошибок")
    cols = st.columns(min(len(models), 3))
    for i, (name, m) in enumerate(models.items()):
        with cols[i % len(cols)]:
            st.pyplot(plot_confusion(y_test, m["pred"], title=name))
