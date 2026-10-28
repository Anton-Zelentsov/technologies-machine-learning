# Лабораторная работа. Анализ и прогнозирование временного ряда

Датасет: **Air Passengers** (Kaggle, [rakannimer/air-passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers)), ежемесячный ряд, 144 наблюдения.
Постановка: https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_TIMESERIES

Методы: SARIMA, символьная регрессия (gplearn), МГУА COMBI и MIA (gmdh).

- `data/AirPassengers.csv` — данные (144 × 2)
- `Lab_TimeSeries.ipynb` / `Lab_TimeSeries.html` / `Lab_TimeSeries.py` — решение (ноутбук выполнен)
- `screenshots/` — скриншоты ячеек ноутбука
- `Отчет временной ряд.docx` — отчёт (титульный лист, задание, текст программы, экранные формы, выводы)
- `build_nb.py`, `build_report.py` — генерация ноутбука и отчёта

Примечание: библиотека `gmdh` собирается из исходников (нужны CMake и Boost); на NumPy 2.x требуется сборка со свежим pybind11 (≥ 2.12).
