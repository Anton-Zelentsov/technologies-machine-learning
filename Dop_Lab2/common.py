"""Общие функции для веб-приложений: данные, метрики, графики."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/breast_cancer.csv"


@st.cache_data
def load_data():
    """Загрузка датасета: удаляем идентификатор и пустой столбец, кодируем целевой признак (M = 1)."""
    df = pd.read_csv(DATA_PATH).drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)
    return df


def split_data(df, test_size, seed):
    X, y = df.drop(columns="diagnosis"), df["diagnosis"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def evaluate(model, X_test, y_test):
    """Метрики качества на тестовой выборке и вероятности положительного класса."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, proba),
    }, pred, proba


def plot_confusion(y_test, pred, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["B", "M"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title or "Матрица ошибок")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинное значение")
    return fig


def plot_roc(y_test, probas):
    """ROC-кривые: probas — словарь {название модели: вероятности}."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend(fontsize=8, loc="lower right")
    return fig
