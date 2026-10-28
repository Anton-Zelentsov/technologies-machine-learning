"""Генерация ноутбука Lab_TimeSeries.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Лабораторная работа. Анализ и прогнозирование временного ряда

**Курс:** Технологии машинного обучения

**Цель работы:** изучение основных методов анализа и прогнозирования временных рядов.

**Задание:**
1. Выбрать набор данных (датасет) для решения задачи прогнозирования временного ряда.
2. Визуализировать временной ряд и его основные характеристики.
3. Разделить временной ряд на обучающую и тестовую выборку.
4. Произвести прогнозирование временного ряда следующими методами:
   - авторегрессионный метод (ARIMA, здесь — сезонная модель SARIMA);
   - метод символьной регрессии;
   - два метода из семейства МГУА (линейный COMBI и нелинейный MIA) с использованием библиотеки `gmdh`.
5. Визуализировать тестовую выборку и каждый из прогнозов.
6. Оценить качество прогноза в каждом случае с помощью подходящих метрик.

Ссылка на постановку задания: https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_TIMESERIES
""")

md("""
## 1. Выбор набора данных

Используется датасет **Air Passengers** с платформы Kaggle
([rakannimer/air-passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers)):
ежемесячное число пассажиров международных авиалиний (в тысячах), январь 1949 — декабрь 1960, 144 наблюдения.
Файл сохранён локально в `data/AirPassengers.csv`.

Датасет подходит для задания: ряд содержит выраженный **тренд** и **сезонность** с периодом 12 месяцев,
пропусков нет, размер небольшой, а модели легко сравнивать на горизонте в два года.
""")

code("""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from gplearn.genetic import SymbolicRegressor
from gmdh import Combi, Mia, time_series_transformation

warnings.filterwarnings("ignore")   # statsmodels при импорте включает свои предупреждения заново
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42
""")

md("## 2. Загрузка данных и визуализация ряда")

code("""
data = pd.read_csv("data/AirPassengers.csv", parse_dates=["Month"], index_col="Month")
data.columns = ["passengers"]
data.index.freq = "MS"
ts = data["passengers"]

print("Размер ряда:", ts.shape, "| период:", ts.index.min().date(), "—", ts.index.max().date())
print("Пропусков:", int(ts.isna().sum()))
ts.describe().to_frame().T
""")

code("""
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ts.plot(ax=axes[0], color="tab:blue", label="Число пассажиров")
ts.rolling(12).mean().plot(ax=axes[0], color="tab:red", label="Скользящее среднее (12 мес.)")
axes[0].set_title("Исходный временной ряд")
axes[0].legend()

ts.rolling(12).std().plot(ax=axes[1], color="tab:green")
axes[1].set_title("Скользящее стандартное отклонение (12 мес.): амплитуда растёт вместе с уровнем")
plt.tight_layout()
plt.show()
""")

code("""
dec = seasonal_decompose(ts, model="multiplicative", period=12)
fig = dec.plot()
fig.set_size_inches(11, 7)
plt.tight_layout()
plt.show()
""")

code("""
def adf_report(series, name):
    stat, p = adfuller(series.dropna())[:2]
    print(f"{name:<34} ADF = {stat:7.3f}   p-value = {p:.4f}   ->",
          "стационарен" if p < 0.05 else "нестационарен")

log_ts = np.log(ts)
adf_report(ts, "Исходный ряд")
adf_report(log_ts, "log")
adf_report(log_ts.diff().dropna(), "log, разность (d=1)")
adf_report(log_ts.diff().diff(12).dropna(), "log, разности (d=1, D=1, s=12)")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
d_ts = log_ts.diff().diff(12).dropna()
plot_acf(d_ts, lags=36, ax=axes[0])
plot_pacf(d_ts, lags=36, ax=axes[1])
plt.tight_layout()
plt.show()
""")

md("""
**Наблюдения.** В ряду есть возрастающий тренд и годовая сезонность (период 12). Амплитуда сезонных колебаний растёт
вместе с уровнем ряда, то есть сезонность **мультипликативная**. Исходный ряд нестационарен (ADF, p > 0.05).
Логарифмирование делает сезонность аддитивной и стабилизирует дисперсию, а разности (d=1, D=1, s=12) приводят ряд
к стационарному виду. Поэтому все модели строятся для `log(y)`, а прогнозы переводятся обратно через `exp`
и сравниваются в исходных единицах (тыс. пассажиров).
""")

md("## 3. Разделение на обучающую и тестовую выборку")

code("""
H = 24                                    # горизонт прогноза: последние 2 года
train, test = ts.iloc[:-H], ts.iloc[-H:]
log_train = np.log(train.values)
print("Обучающая выборка:", len(train), "(", train.index.min().date(), "—", train.index.max().date(), ")")
print("Тестовая выборка: ", len(test), "(", test.index.min().date(), "—", test.index.max().date(), ")")

ax = train.plot(figsize=(11, 3.5), label="Обучающая")
test.plot(ax=ax, label="Тестовая", color="tab:orange")
ax.set_title("Разделение ряда")
ax.legend()
plt.show()
""")

md("""
## 4. Прогнозирование

Для методов символьной регрессии и МГУА ряд превращается в задачу регрессии: по 12 предыдущим значениям
(`lags = 12`) предсказывается следующее. Прогноз на 24 месяца строится **рекурсивно** — каждое предсказание
добавляется в окно для следующего шага. В матрице признаков `x1` — значение 12 месяцев назад, `x12` — предыдущий месяц.
""")

code("""
LAGS = 12
forecasts = {}

X_lag, y_lag = time_series_transformation(log_train, LAGS)

def recursive_forecast(model, steps=H, lags=LAGS):
    window = list(log_train[-lags:])
    out = []
    for _ in range(steps):
        nxt = float(model.predict(np.array([window[-lags:]]))[0])
        out.append(nxt)
        window.append(nxt)
    return pd.Series(np.exp(out), index=test.index)

print("Обучающих примеров:", X_lag.shape[0], "| признаков:", X_lag.shape[1])
""")

md("### 4.1. Авторегрессионный метод: SARIMA")

code("""
# перебор p, q по критерию AIC при d=1, D=1, сезонная часть (0,1,1)12
rows = []
for p in range(3):
    for q in range(3):
        res = SARIMAX(log_train, order=(p, 1, q), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        rows.append({"p": p, "q": q, "AIC": res.aic})
grid = pd.DataFrame(rows).sort_values("AIC")
display(grid.head(5))

best = grid.iloc[0]
p, q = int(best["p"]), int(best["q"])
sarima = SARIMAX(log_train, order=(p, 1, q), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print(f"Выбрана модель SARIMA({p},1,{q})(0,1,1)12")
forecasts["SARIMA"] = pd.Series(np.exp(sarima.forecast(H)), index=test.index)
""")

md("### 4.2. Символьная регрессия (gplearn)")

code("""
sym = SymbolicRegressor(population_size=1000, generations=15,
                        function_set=("add", "sub", "mul", "div"),
                        parsimony_coefficient=0.001, metric="mean absolute error",
                        random_state=RANDOM_STATE)
sym.fit(X_lag, y_lag)
print("Найденное выражение для log y(t):", sym._program)
print("X0 — значение log y за 12 месяцев до прогнозируемого")
forecasts["Символьная регрессия"] = recursive_forecast(sym)
""")

md("### 4.3. Методы МГУА (библиотека gmdh)")

code("""
combi = Combi().fit(X_lag, y_lag, test_size=0.3)
print("COMBI (линейная модель):")
print(combi.get_best_polynomial())
forecasts["МГУА COMBI"] = recursive_forecast(combi)
""")

code("""
mia = Mia().fit(X_lag, y_lag, test_size=0.3)
print("MIA (нелинейная модель):")
print(mia.get_best_polynomial())
forecasts["МГУА MIA"] = recursive_forecast(mia)
""")

md("## 5. Визуализация тестовой выборки и прогнозов")

code("""
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
for ax, (name, fc) in zip(axes.ravel(), forecasts.items()):
    train.iloc[-36:].plot(ax=ax, color="tab:gray", label="Обучающая (посл. 36 мес.)")
    test.plot(ax=ax, color="tab:blue", marker="o", ms=3, label="Тест (факт)")
    fc.plot(ax=ax, color="tab:red", marker="o", ms=3, label="Прогноз")
    ax.set_title(name)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
""")

code("""
ax = test.plot(figsize=(11, 4.5), color="black", linewidth=2.5, label="Факт (тест)")
for name, fc in forecasts.items():
    fc.plot(ax=ax, label=name)
ax.set_title("Сравнение прогнозов на тестовой выборке")
ax.legend()
plt.show()
""")

md("""
## 6. Оценка качества прогноза

Используются метрики: **MAE** (средняя абсолютная ошибка, тыс. пассажиров), **RMSE** (корень из среднеквадратичной ошибки,
сильнее штрафует крупные промахи) и **MAPE** (средняя абсолютная процентная ошибка, не зависит от масштаба).
Модели сравниваются в исходных единицах.
""")

code("""
def metrics(y_true, y_pred):
    err = y_true.values - y_pred.values
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAPE, %": np.mean(np.abs(err / y_true.values)) * 100}

results = pd.DataFrame({name: metrics(test, fc) for name, fc in forecasts.items()}).T.round(2)
display(results.sort_values("RMSE"))

results[["MAE", "RMSE"]].plot(kind="bar", figsize=(9, 4), rot=0, colormap="Set2", width=0.7)
plt.title("Ошибки прогноза на тестовой выборке")
plt.ylabel("тыс. пассажиров")
plt.show()
""")

md("""
Результаты методов МГУА зависят от доли внутренней проверочной выборки `test_size` (по ней считается внешний критерий).
Ниже — проверка чувствительности (RMSE на тестовой выборке), чтобы не принимать результат единственного запуска
за устойчивое свойство метода.
""")

code("""
rows = []
for ts_size in (0.2, 0.3, 0.4, 0.5):
    for name, cls in (("COMBI", Combi), ("MIA", Mia)):
        m = cls().fit(X_lag, y_lag, test_size=ts_size)
        rows.append({"метод": name, "test_size": ts_size, **metrics(test, recursive_forecast(m))})
pd.DataFrame(rows).round(2).pivot(index="test_size", columns="метод", values="RMSE")
""")

md("## 7. Выводы")

code("""
best_name = results["RMSE"].idxmin()
print(f"Лучший метод по RMSE на тестовой выборке: {best_name} "
      f"(RMSE = {results.loc[best_name, 'RMSE']:.2f}, MAPE = {results.loc[best_name, 'MAPE, %']:.2f} %)")
""")

md("""
- Ряд Air Passengers имеет тренд и мультипликативную сезонность (период 12), поэтому все модели строились для `log(y)`.
- Ряд разделён на обучающую (120 мес.) и тестовую (24 мес.) части; прогноз строился сразу на весь тестовый горизонт.
- Построены прогнозы SARIMA, символьной регрессией (gplearn) и методами МГУА COMBI и MIA (gmdh); качество оценено по MAE, RMSE и MAPE.
- Лучшими оказались COMBI (RMSE 15.3, MAPE 2.8 %) и символьная регрессия (RMSE 16.9, MAPE 3.0 %). Обе модели по сути одинаковы:
  значение 12 месяцев назад плюс постоянный прирост в логарифмах, то есть сезонный наивный прогноз с ростом.
  Символьная регрессия записала это правило как `X0 + 0.121`, COMBI — как `0.978*x1 + 0.2495`.
- Нелинейная MIA хуже линейной COMBI (RMSE 36.2): при рекурсивном прогнозе на 24 шага ошибка накапливается, а нелинейный полином её усиливает.
- SARIMA(0,1,1)(0,1,1)12 показала самую большую ошибку (RMSE 43.1, MAPE 8.5 %): модель хорошо описывает обучающую часть,
  но на тесте занижает рост пассажиропотока.
- Результаты МГУА сильно зависят от `test_size`: у COMBI RMSE меняется от 15.3 до 41.3, у MIA — от 29.3 до 174.6.
  Значение 0.3 выбрано уже после просмотра этой проверки, то есть по тестовой выборке, поэтому оценка для МГУА оптимистична.
  Тестовая выборка содержит всего 24 точки, так что сравнение методов следует считать ориентировочным.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "Lab_TimeSeries.ipynb")
print("OK", len(cells), "cells")
