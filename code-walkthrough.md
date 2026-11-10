# Как работает код лабораторной № 1

```text
CSV → очистка → train / val / test → массивы .npz
                                        ↓
                  обучение сетей → сравнение → графики и метрики
```

| Инструмент | Что делает в проекте |
| --- | --- |
| pandas | Читает CSV, выбирает столбцы, группирует строки и убирает повторы |
| scikit-learn | Делит данные, готовит признаки, считает метрики и строит базовые прогнозы |
| NumPy | Хранит подготовленные числа в массивах и сжимает их в `.npz` |
| PyTorch | Собирает сеть, считает градиенты и обновляет веса |
| Matplotlib | Сохраняет графики обучения, ROC-кривые и матрицы ошибок |

## 1. Что подаём на вход и что предсказываем

В [prepare_data.py](prepare_data.py) словарь `DATASETS` описывает каждый CSV. Функция `clean_data()` оставляет нужные столбцы, переводит цели в нужный формат и удаляет точные повторы.

```python
features = list(dataset.numeric + dataset.categorical + dataset.binary)
data = raw.loc[:, [*features, dataset.target]].copy()
if dataset.labels is not None:
    data[dataset.target] = data[dataset.target].map(dataset.labels)
```

| Набор | Целевой столбец | Что делает код с ответом |
| --- | --- | --- |
| Diabetes | `Diabetes_012` | Оставляет `0`, объединяет `1` и `2` в класс `1` |
| Body Performance | `class` | Превращает A, B, C, D в 0, 1, 2, 3 |
| Wine Quality | `quality` | Оставляет оценку вина в баллах |

Если в Body Performance одно из значений давления неположительное или диастолическое выше систолического, код помечает оба числа пропущенными. Таких строк пять. Числа для заполнения берутся позже из обучающей части.

## 2. Как делим данные

`split_data()` объединяет строки с одинаковыми входами в один профиль. Это удерживает одинаковые входы в одной части, даже если у строк разные ответы.

```python
groups = features.groupby(list(features.columns), sort=False, dropna=False).ngroup()
profiles = groups.drop_duplicates().to_numpy()
```

```text
Все профили
├── 70% → train → обновляем веса
└── 30% → отложенные профили
    ├── половина → val  → выбираем настройки и модель
    └── половина → test → измеряем итоговое качество
```

```python
train, holdout = train_test_split(
    profiles, test_size=0.30, random_state=RANDOM_STATE,
    stratify=strata.loc[profiles] if strata is not None else None,
)
validation, test = train_test_split(
    holdout, test_size=0.50, random_state=RANDOM_STATE,
    stratify=strata.loc[holdout] if strata is not None else None,
)
```

Для классификации `stratify` помогает сохранить пропорции меток между частями. Для регрессии код делит профили случайно. `RANDOM_STATE = 42` позволяет повторить то же разбиение. Проценты относятся к профилям, в одном профиле может быть несколько строк.

| Набор | Train, строк | Val, строк | Test, строк | Входов сети |
| --- | ---: | ---: | ---: | ---: |
| Diabetes | 160 798 | 34 456 | 34 458 | 49 |
| Body Performance | 9 374 | 2 009 | 2 009 | 12 |
| Wine Quality | 2 772 | 594 | 595 | 11 |

## 3. Как приводим признаки к числам

`make_preprocessor()` собирает преобразования, а `prepare_dataset()` применяет их к каждой части.

| Что пришло из CSV | Что делает инструмент | Зачем |
| --- | --- | --- |
| Число, например BMI | `SimpleImputer` заполняет пропуск медианой, `StandardScaler` меняет масштаб | Признаки в разных единицах получают сопоставимый масштаб |
| Категория, например `gender` | `SimpleImputer` берёт самое частое значение, `OneHotEncoder` создаёт столбцы 0/1 | Сеть получает числа, а категории сохраняют свой смысл |
| Готовый признак 0/1 | `SimpleImputer` заполняет пропуск самым частым значением | Код 0/1 уже подходит для входа сети |

```python
splits = split_data(features, target, stratify=classification)
preprocessor = make_preprocessor(dataset)
preprocessor.fit(features.loc[splits["train"]])
```

```python
result[f"X_{name}"] = preprocessor.transform(features.loc[index]).astype(np.float32)
result[f"y_{name}"] = target.loc[index].to_numpy(
    dtype=np.int64 if classification else np.float32
)
```

```text
train ── fit() ──→ медианы, масштабы и список категорий
                              │
                              └── transform() для train, val и test
```

Если вычислить медианы или масштабы по всему CSV, сведения из теста попадут в подготовку данных. Поэтому `fit()` видит только `train`. В архиве NumPy `X_*` хранит входы сети, `y_*` — правильные ответы, а `rows_*` — номера исходных строк.

## 4. Как собирается сеть

В [train_models.py](train_models.py) функция `build_model()` собирает полносвязную сеть, или MLP. `nn.Linear` смешивает входные числа с помощью весов, ReLU или Tanh добавляет нелинейность, а последний слой выдаёт ответ нужного размера.

```python
activation = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}[architecture.activation]
for width in architecture.widths:
    layers.extend((nn.Linear(inputs, width), activation()))
    if architecture.dropout:
        layers.append(nn.Dropout(architecture.dropout))
    inputs = width
return nn.Sequential(*layers, nn.Linear(inputs, outputs))
```

```text
Body Performance, MLP-3
12 входов → 128 нейронов → 64 нейрона → 32 нейрона → 4 оценки классов
```

| Модель | Скрытые слои | Что проверяем |
| --- | --- | --- |
| MLP-1 | 32 | Компактная сеть |
| MLP-2 | 64, 32 | Сеть среднего размера с ReLU и Adam |
| MLP-3 | 128, 64, 32 | Более широкая и глубокая сеть |
| MLP-2-tanh | 64, 32 | Меняем ReLU на Tanh |
| MLP-2-rmsprop | 64, 32 | Меняем Adam на RMSprop |
| MLP-2-dropout | 64, 32 | Добавляем Dropout 0,15 после каждого скрытого слоя |

Adam и RMSprop задают, как обновлять веса по накопленным градиентам. Dropout случайно обнуляет часть значений в скрытых слоях во время обучения. Так можно проверить, помогает ли сети меньше зависеть от отдельных нейронов.

## 5. Как сеть узнаёт, насколько ошиблась

```python
if task == "binary":
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
elif task == "multiclass":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()
```

| Набор | Как код считает ошибку | Как получаем прогноз |
| --- | --- | --- |
| Diabetes | `BCEWithLogitsLoss` сравнивает один выход с ответом 0 или 1 | `sigmoid` переводит выход в число от 0 до 1, затем применяем порог |
| Body Performance | `CrossEntropyLoss` сравнивает четыре выхода с номером верного класса | `softmax` даёт четыре оценки с суммой 1, выбираем максимальную |
| Wine Quality | `MSELoss` считает средний квадрат разницы между прогнозом и оценкой | Берём число с выхода сети как оценку в баллах |

В Diabetes положительных примеров меньше. `pos_weight` даёт их ошибкам вес около 4,79, чтобы сеть сильнее учитывала этот класс. Число после `sigmoid` используется для ранжирования и решения по порогу. Насколько оно совпадает с реальной частотой диабета, в работе отдельно не проверяли.

## 6. Что происходит на каждой эпохе

Эпоха — один проход по обучающим строкам. Код перемешивает их и делит на пакеты по 1 024 строки для Diabetes и по 128 для двух других наборов.

```python
indices = torch.randperm(len(X_train), generator=batch_generator)
for batch in indices.split(batch_size):
    optimizer.zero_grad()
    loss = criterion(model(X_train[batch]), y_train[batch])
    loss.backward()
    optimizer.step()
```

| Строка кода | Простыми словами |
| --- | --- |
| `zero_grad()` | Стираем градиенты предыдущего пакета |
| `model(X_train[batch])` | Получаем прогноз для строк пакета |
| `criterion(...)` | Измеряем ошибку прогноза |
| `loss.backward()` | Считаем, как изменение каждого веса повлияет на ошибку |
| `optimizer.step()` | Обновляем веса |

```text
Пакет → прогноз → ошибка → градиенты → новые веса
Конец эпохи → train loss и val loss → вывод в терминал
```

После эпохи `model.eval()` отключает Dropout, а `torch.inference_mode()` позволяет посчитать ошибки без градиентов. Ошибки на `train` и `val` рассчитываются с одними и теми же весами.

## 7. Когда останавливаемся и как выбираем победителя

```python
if val_loss < best_loss - MIN_DELTA:
    best_loss = val_loss
    best_epoch = epoch
    best_state = deepcopy(model.state_dict())
    stale = 0
else:
    stale += 1
```

Если `val loss` стала меньше более чем на 0,00001, скрипт сохраняет копию весов и сбрасывает счётчик ожидания. Если шесть эпох подряд улучшения нет, скрипт заканчивает обучение и восстанавливает сохранённые веса. Лимит — 40 эпох.

```text
Body Performance, MLP-3
24-я эпоха → сохранили веса
25–30-я    → улучшения на val не было
остановка  → вернулись к весам 24-й эпохи
```

За эти шесть эпох `train loss` снизилась с 0,5624 до 0,5476, а `val loss` выросла с 0,6124 до 0,6173. Сеть лучше подстроилась под обучающие данные, но хуже показала себя на валидации.

```python
metric = "mse" if task == "regression" else "balanced_accuracy"
direction = min if task == "regression" else max
winner = direction(runs, key=lambda label: runs[label]["validation"][metric])
```

| Какое решение | На каких данных | По какому правилу |
| --- | --- | --- |
| Какие веса оставить у каждой сети | Val | Меньше `val loss` |
| Где провести границу между 0 и 1 у Diabetes | Val | Выбрать порог с максимальной Balanced Accuracy среди проверенных порогов |
| Какую сеть выбрать | Val | Выше Balanced Accuracy у классификаторов, ниже MSE у вина |
| Как измерить итоговое качество | Test | Посчитать метрики после того, как модель и порог выбраны |

Для Diabetes функция `choose_threshold()` смотрит на ROC-кривую. Она выбирает порог, где разница между долей найденных положительных и долей ложных тревог максимальна. При равенстве берёт порог ближе к 0,5.

## 8. Что получилось и почему

Базовый классификатор берёт только доли классов в `train`. Базовый прогноз для вина выдаёт среднюю оценку из `train`. Это ориентиры, с которыми сравниваются сети.

| Набор | Победитель по val | Что получили на test | Базовый результат |
| --- | --- | --- | --- |
| Diabetes | MLP-2-tanh | Balanced Accuracy **0,7384**, ROC-AUC **0,8132** | Balanced Accuracy 0,5000, ROC-AUC 0,5000 |
| Body Performance | MLP-3 | Balanced Accuracy **0,7551**, средняя ROC-AUC **0,9233** | Balanced Accuracy 0,2500, средняя ROC-AUC 0,5000 |
| Wine Quality | MLP-2-tanh | MSE **0,5518**, MAE **0,5783**, R² **0,2972** | MSE 0,7860, MAE 0,6719, R² −0,0012 |

| Что видно | Как это объяснить |
| --- | --- |
| **Diabetes.** Recall 0,8045 и Precision 0,3387 | При пороге 0,4823 сеть нашла 4 786 из 5 949 положительных случаев. Она также дала 9 344 ложных срабатывания. Поэтому доля верных положительных прогнозов гораздо ниже доли найденных случаев |
| **Body Performance.** Труднее всего найти класс B | Из 502 объектов B сеть угадала 322, а 119 отнесла к A. Высокая ROC-AUC описывает порядок оценок, а Balanced Accuracy — уже итоговый выбор класса |
| **Wine Quality.** Ошибка ниже базовой, но оценки предсказываются с разбросом | MSE примерно на 30% ниже прогноза средним. Абсолютная ошибка в среднем составляет 0,5783 балла. Все пять примеров с оценкой 9 попали в train, поэтому тест охватывает оценки 3–8 |

| Термин | Что означает |
| --- | --- |
| Recall | Какую долю настоящих объектов класса мы нашли |
| Precision | Какая доля прогнозов этого класса оказалась верной |
| Balanced Accuracy | Среднее значение Recall по классам. Редкий и частый класс имеют одинаковый вес |
| ROC-AUC | Насколько хорошо модель ставит нужный класс выше остальных по своим оценкам |
| MAE | Средняя величина ошибки в баллах |
| MSE | Средний квадрат ошибки. Большие промахи влияют сильнее |
| R² | Насколько меньше квадрат ошибки по сравнению с разбросом оценок вокруг их среднего |

Для вина конфигурация с RMSprop получила чуть меньшую MSE на тесте, но по валидации победила Tanh. Результаты получены на одном разбиении, поэтому порядок моделей может измениться на другом наборе.

## 9. Где лежат результаты

| Файл | Что искать |
| --- | --- |
| [metrics.json](results/metrics.json) | Потери по эпохам, выбранные модели, пороги, метрики и матрицы ошибок |
| [diabetes_roc.png](results/diabetes_roc.png) | Как меняются полнота и доля ложных срабатываний при разных порогах |
| [body_performance_confusion.png](results/body_performance_confusion.png) | Какие классы чаще путаются. Строка — верный класс, столбец — прогноз |
| [wine_quality_loss.png](results/wine_quality_loss.png) | Как менялась ошибка на train и val |
| [report.md](report.md) | Полные таблицы и выводы |
| Локальные `results/*.pt` | Веса выбранной эпохи каждой конфигурации |

Запуск из корня проекта.

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe train_models.py
```
