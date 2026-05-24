from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = Path(os.getenv("LANDSCAPE_PROJECT_DIR", Path(__file__).parents[1]))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

DEFAULT_ARGS = {
    "owner": "landscape-classifier",
    "depends_on_past": False,
    "retries": 0,
}


def project_command(command: str) -> str:
    return f'cd "{PROJECT_DIR}" && {command}'


with DAG(
    dag_id="landscape_training_pipeline",
    description="Train, register, export and prepare Triton repository",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["landscape", "training", "mlflow", "triton"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=project_command(
            "test -d data/seg_train/seg_train && "
            "test -d data/seg_test/seg_test || "
            "uv run dvc pull data.dvc"
        ),
    )

    train = BashOperator(
        task_id="train",
        bash_command=project_command(
            f"uv run landscape-train mlflow.tracking_uri={MLFLOW_URI}"
        ),
    )

    export_onnx = BashOperator(
        task_id="export_onnx",
        bash_command=project_command(
            f"uv run landscape-export-onnx mlflow.tracking_uri={MLFLOW_URI}"
        ),
    )

    prepare_triton_repository = BashOperator(
        task_id="prepare_triton_repository",
        bash_command=project_command(
            f"uv run landscape-prepare-triton mlflow.tracking_uri={MLFLOW_URI}"
        ),
    )

    prepare_data >> train >> export_onnx >> prepare_triton_repository
