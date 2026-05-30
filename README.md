# Nature's Palette: классификатор ландшафтов

Проект решает задачу многоклассовой классификации изображений ландшафтов.
Модель принимает RGB-изображение в формате JPEG/PNG и возвращает идентификатор
класса, название класса и confidence score.

Поддерживаемые классы:
`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

В проекте используется датасет Intel Image Classification, ResNet18, PyTorch
Lightning, Hydra, MLflow, DVC, ONNX и Triton Inference Server.

Основной production flow:

data -> training -> MLflow Registry -> production alias
-> ONNX export -> Triton model repository -> Triton server -> FastAPI

## Цель проекта

Модель может использоваться для сортировки медиа, визуального поиска,
каталогизации изображений и базовой модерации визуального контента. Основной
production-путь устроен так: обучение регистрирует модель в MLflow, alias
`production` выбирает рабочую версию, затем эта версия экспортируется в ONNX и
обслуживается через Triton.

## Стек

- Python 3.11
- uv для управления зависимостями и lock-файлом
- PyTorch, Torchvision, PyTorch Lightning
- Hydra для конфигурации
- DVC для данных и модельных артефактов
- MLflow для экспериментов и Model Registry
- ONNX для production packaging
- Triton Inference Server для production inference
- FastAPI как HTTP API поверх Triton
- pre-commit, ruff, prettier для проверки качества кода

## Setup

Склонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/<your-github-username>/landscape-classifier.git
cd landscape-classifier
uv sync
```

Установите и запустите pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run -a
```

Запустите MLflow на адресе, который ожидается в задании:

```bash
docker compose up -d mlflow
```

MLflow UI:

```text
http://127.0.0.1:8080
```

Получите данные через DVC после чистого клонирования:

```bash
uv run dvc pull data.dvc
```

Команды обучения и инференса также пытаются подтянуть нужные DVC-артефакты,
если ожидаемые файлы отсутствуют локально.

## Train

Обучение конфигурируется через Hydra. Главная точка входа в конфиги:
`configs/config.yaml`. Группы конфигов лежат в `configs/`.

Запуск обучения:

```bash
uv run landscape-train
```

Показать resolved Hydra config без запуска обучения:

```bash
uv run landscape-train --cfg job
```

Переопределить параметры из CLI:

```bash
uv run landscape-train training.epochs=1 training.batch_size=32
```

После обучения создаются:

- `models/best_resnet18.pt`
- `artifacts/`
- `plots/training_curves.png`
- MLflow run с параметрами, метриками, графиками и артефактами
- новая версия модели в MLflow Model Registry

После успешного обучения команда автоматически обновляет MLflow alias:

```text
models:/landscape-resnet18@production
```

Проверить, на какую версию модели указывает alias `production`:

```bash
uv run python -c "from mlflow import MlflowClient; c=MlflowClient('http://127.0.0.1:8080'); v=c.get_model_version_by_alias('landscape-resnet18', 'production'); print('name=', v.name); print('version=', v.version); print('run_id=', v.run_id); print('status=', v.status)"
```

Этот alias является источником production-версии. После переобучения нужно
обновить Triton repository:

```bash
uv run landscape-prepare-triton
```

Эта команда загружает `models:/landscape-resnet18@production` из MLflow,
экспортирует production-модель в ONNX и обновляет
`triton_model_repository/`.

## DVC Pipeline

Запуск всего воспроизводимого пайплайна:

```bash
uv run dvc repro
```

Стадии пайплайна:

- `prepare_data`: подтягивает данные через DVC
- `train`: обучает Lightning-модель
- `export_onnx`: экспортирует production-модель в ONNX
- `prepare_triton_repo`: готовит Triton model repository

## Production Preparation

Экспорт текущей MLflow production-модели в ONNX:

```bash
uv run landscape-export-onnx
```

Результат:

```text
models/best_resnet18.onnx
```

Подготовка Triton model repository из MLflow production alias:

```bash
uv run landscape-prepare-triton
```

Результат:

```text
triton_model_repository/landscape_classifier/
├── 1/
│   └── model.onnx
└── config.pbtxt
```

TensorRT-конвертация намеренно не входит в текущую сдаваемую версию. Production
packaging в этом проекте выполнен через ONNX.

## Infer

### Triton Inference Server

Подготовьте ONNX и Triton repository:

```bash
uv run landscape-prepare-triton
```

Запустите Triton:

```bash
docker compose up -d triton
```

Проверка через Python Triton client:

```bash
uv run landscape-triton-predict "data/seg_test/seg_test/forest/20057.jpg"
```

### FastAPI поверх Triton

FastAPI не загружает PyTorch-модель напрямую. Он принимает изображение,
применяет тот же preprocessing, отправляет tensor в Triton и возвращает ответ
пользователю.

Локальный запуск API:

```bash
uv run landscape-api
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Пример запроса:

```bash
curl -X POST \
  -F "file=@data/seg_test/seg_test/forest/20057.jpg" \
  http://127.0.0.1:8000/predict
```

Пример ответа:

```json
{
  "class_id": 1,
  "class_name": "forest",
  "confidence": 0.9912
}
```

Запуск FastAPI и Triton через Docker Compose:

```bash
docker compose up -d triton api
```

В compose API обращается к Triton по внутреннему адресу `triton:8000`.
При локальном запуске `uv run landscape-api` используется значение из Hydra:
`localhost:8001`.

## Code Quality

Проверки качества кода:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run -a
```

## Структура проекта

```text
.
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── export/
│   ├── inference/
│   ├── mlflow/
│   ├── model/
│   ├── training/
│   └── triton/
├── landscape_classifier/
│   ├── commands.py
│   ├── config.py
│   ├── dvc_utils.py
│   ├── export.py
│   ├── inference/
│   ├── training/
│   └── triton.py
├── tests/
├── triton_model_repository/
├── data.dvc
├── dvc.yaml
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```
