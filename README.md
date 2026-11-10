# Лабораторная работа № 1. Полносвязные нейронные сети

Нечётный вариант. В проекте сравниваются архитектуры MLP и базовые модели для бинарной классификации диабета, классификации физической подготовки и регрессии качества белого вина.

## Запуск в Windows

Используйте Python 3.13 и `uv`. Поместите следующие файлы в папку `data/raw`, сохранив их имена.

| Файл | Источник |
| --- | --- |
| `diabetes_012_health_indicators_BRFSS2015.csv` | [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| `bodyPerformance.csv` | [Body Performance](https://www.kaggle.com/datasets/kukuroo3/body-performance-data) |
| `winequality-white.csv` | [Wine Quality, UCI](https://archive.ics.uci.edu/dataset/186/wine%2Bquality) |

Из корня проекта выполните:

```powershell
uv sync --locked
uv run --locked python prepare_data.py
uv run --locked python train_models.py
```

Скрипты создают обработанные данные в `data/processed` и сохраняют веса, метрики и графики в `results`. Исходные и обработанные датасеты, окружение и веса остаются локальными.

В [отчёте](report.md) приведены методика и полные таблицы. Для защиты есть [презентация PowerPoint](deliverables/lab-1-defense.pptx). Метрики и графики последнего эксперимента доступны в [results](results/).
