# Лабораторная работа. Веб-приложение для демонстрации моделей машинного обучения

Датасет: **Breast Cancer Wisconsin** (Kaggle, [uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)), бинарная классификация (M / B).
Постановка: https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO_WEB
Фреймворк: Streamlit.

- `data/breast_cancer.csv` — данные (569 × 33)
- `common.py` — общие функции: загрузка данных, метрики, графики
- `app_variant1.py` — вариант 1: одна модель (случайный лес), настройка гиперпараметров
- `app_variant2.py` — вариант 2: выбор и сравнение нескольких моделей
- `screenshots/` — скриншоты работающих приложений
- `Отчет веб-приложение.docx` — отчёт (титульный лист, задание, вариант 1, вариант 2, выводы)
- `take_screenshots.py`, `build_report.py` — съёмка скриншотов и сборка отчёта

## Запуск (из этой папки)

```
pip install -r requirements.txt
streamlit run app_variant1.py
streamlit run app_variant2.py
```
