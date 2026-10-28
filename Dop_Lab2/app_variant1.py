"""Вариант 1. Веб-приложение для одной модели машинного обучения (случайный лес).

Запуск: streamlit run app_variant1.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from common import load_data, split_data, evaluate, plot_confusion, plot_roc

st.set_page_config(page_title="Вариант 1: случайный лес", layout="wide")
st.title("Вариант 1. Случайный лес: диагностика опухоли")
st.caption("Датасет: Breast Cancer Wisconsin (Kaggle). Задача: определить, злокачественная ли опухоль (M) или доброкачественная (B).")

df = load_data()

# ---- гиперпараметры алгоритма ----
st.sidebar.header("Гиперпараметры")
n_estimators = st.sidebar.slider("n_estimators (число деревьев)", 10, 500, 100, step=10)
limit_depth = st.sidebar.checkbox("Ограничить max_depth", value=True)
max_depth = st.sidebar.slider("max_depth", 1, 20, 5) if limit_depth else None
min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2)
max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", None], format_func=str)
st.sidebar.header("Разбиение выборки")
test_size = st.sidebar.slider("Доля тестовой выборки", 0.1, 0.5, 0.25, step=0.05)
seed = st.sidebar.number_input("random_state", 0, 9999, 42)

with st.expander("Данные", expanded=False):
    st.write(f"Размер: {df.shape[0]} объектов, {df.shape[1] - 1} признаков. "
             f"Злокачественных: {int(df['diagnosis'].sum())}, доброкачественных: {int((df['diagnosis'] == 0).sum())}.")
    st.dataframe(df.head(10))

# ---- обучение ----
if st.sidebar.button("Обучить модель", type="primary"):
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   max_features=max_features, random_state=seed)
    model.fit(X_train, y_train)
    metrics, pred, proba = evaluate(model, X_test, y_test)
    st.session_state["result"] = {
        "metrics": metrics, "pred": pred, "proba": proba, "y_test": y_test,
        "train_acc": model.score(X_train, y_train),
        "importances": pd.Series(model.feature_importances_, index=X_train.columns).sort_values(),
        "params": dict(n_estimators=n_estimators, max_depth=max_depth,
                       min_samples_split=min_samples_split, max_features=max_features),
    }

# ---- просмотр результатов ----
res = st.session_state.get("result")
if res is None:
    st.info("Задайте гиперпараметры в боковой панели и нажмите «Обучить модель».")
else:
    st.subheader("Результаты обучения")
    st.write("Параметры модели:", res["params"])
    cols = st.columns(6)
    cols[0].metric("Accuracy (train)", f"{res['train_acc']:.3f}")
    for col, (name, value) in zip(cols[1:], res["metrics"].items()):
        col.metric(f"{name} (test)", f"{value:.3f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(plot_confusion(res["y_test"], res["pred"]))
    with c2:
        st.pyplot(plot_roc(res["y_test"], {"Случайный лес": res["proba"]}))
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        res["importances"].tail(10).plot.barh(ax=ax, color="tab:green")
        ax.set_title("Важность признаков (топ-10)")
        plt.tight_layout()
        st.pyplot(fig)
