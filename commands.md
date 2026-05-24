# Commands

## Setup

```bash
uv sync
uv run pre-commit install
uv run pre-commit run -a
```

## MLflow

```bash
docker compose up -d mlflow
```

MLflow UI:

```text
http://127.0.0.1:8080
```

Stop services:

```bash
docker compose down
```

## DVC

```bash
uv run dvc pull data.dvc
uv run dvc repro
```

## Airflow

```bash
docker compose up -d mlflow
```

```bash
AIRFLOW_HOME="$PWD/airflow" \
AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags" \
AIRFLOW__CORE__LOAD_EXAMPLES=false \
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true \
uv run --with "apache-airflow==2.9.3" airflow standalone
```

Airflow UI:

```text
http://127.0.0.1:8081
```

Credentials:

```text
admin / admin
```

Trigger DAG from CLI:

```bash
AIRFLOW_HOME="$PWD/airflow" \
AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags" \
uv run --with "apache-airflow==2.9.3" airflow dags trigger landscape_training_pipeline
```

## Training

```bash
uv run landscape-train
```

Show resolved Hydra config:

```bash
uv run landscape-train --cfg job
```

Short smoke run:

```bash
uv run landscape-train training.epochs=1
```

## ONNX Export

Exports `models:/landscape-resnet18@production` from MLflow Registry.

```bash
uv run landscape-export-onnx
```

## Triton

Prepare repository:

```bash
uv run landscape-prepare-triton
```

Run Triton:

```bash
docker compose up -d triton
```

Predict through Triton:

```bash
uv run landscape-triton-predict "data/seg_test/seg_test/forest/20057.jpg"
```

## FastAPI

FastAPI uses Triton as the inference backend.

```bash
uv run landscape-prepare-triton
docker compose up -d triton
```

```bash
uv run landscape-api
```

```bash
curl -X POST \
  -F "file=@data/seg_test/seg_test/forest/20057.jpg" \
  http://127.0.0.1:8000/predict
```

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run -a
```
