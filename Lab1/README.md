# Лабораторная работа №1. EDA и визуализация данных

Разведочный анализ данных (EDA) на датасете **Wine recognition dataset** из Scikit-learn.

## Задание

По методичке курса ТМО (лаба 1 — EDA/Visualization):
- выбрать датасет без пропусков (взят встроенный датасет Scikit-learn `load_wine`);
- сделать ноутбук с текстовым описанием датасета, основными характеристиками, визуальным исследованием и информацией о корреляции признаков;
- разместить отчёт в репозитории на GitHub.

Ссылки на задание:
- https://github.com/ugapanyuk/courses_current/wiki/LAB_TMO__EDA_VISUALIZATION
- Список датасетов: https://github.com/ugapanyuk/courses_current/wiki/DSLIST
- Пример преобразования Scikit-learn → Pandas: https://github.com/ugapanyuk/courses_current/blob/main/notebooks/ds/sklearn_datasets.ipynb

## Содержимое

- `EDA_Lab1.ipynb` — основной ноутбук с решением (выполнен, все ячейки с выводом).
- `EDA_Lab1.html` — экспорт ноутбука в HTML для просмотра без Jupyter.
- `requirements.txt` — зависимости для воспроизведения.

## Как запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook EDA_Lab1.ipynb
```

## Краткое содержание ноутбука

1. Текстовое описание датасета (Wine recognition dataset, 178 наблюдений, 13 признаков, 3 класса).
2. Основные характеристики: размер, типы данных, проверка пропусков, describe(), баланс классов.
3. Визуальное исследование: распределение классов, гистограммы признаков, boxplot по классам, pairplot ключевых признаков.
4. Корреляционный анализ: heatmap корреляций Пирсона, топ-10 самых сильных попарных корреляций, выводы.
