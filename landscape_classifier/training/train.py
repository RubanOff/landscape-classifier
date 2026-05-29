import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from landscape_classifier.config import get_project_root
from landscape_classifier.training.artifacts import (
    save_class_balance,
    save_class_distribution,
    save_dataset_stats,
    save_misclassified_examples,
    save_model_info,
    save_model_summary,
    save_model_tester_artifacts,
    save_sample_images,
    save_training_curves,
)
from landscape_classifier.training.config import save_resolved_config
from landscape_classifier.training.dataset import build_datamodule
from landscape_classifier.training.evaluate import (
    evaluate,
    get_classification_report,
    get_confusion_matrix,
)
from landscape_classifier.training.lit_model import LandscapeClassifierModule
from landscape_classifier.training.logger import get_logger

logger = get_logger()
PROJECT_ROOT = get_project_root()


class MetricHistoryCallback(pl.Callback):
    def __init__(self):
        self.history: list[dict[str, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        self.history.append(
            {
                "epoch": trainer.current_epoch + 1,
                "train_loss": _metric_value(metrics.get("train_loss")),
                "train_accuracy": _metric_value(metrics.get("train_accuracy")),
                "train_weighted_f1": _metric_value(metrics.get("train_weighted_f1")),
                "train_weighted_precision": _metric_value(
                    metrics.get("train_weighted_precision")
                ),
                "train_weighted_recall": _metric_value(
                    metrics.get("train_weighted_recall")
                ),
                "val_loss": _metric_value(metrics.get("val_loss")),
                "val_accuracy": _metric_value(metrics.get("val_accuracy")),
                "val_weighted_f1": _metric_value(metrics.get("val_weighted_f1")),
                "val_weighted_precision": _metric_value(
                    metrics.get("val_weighted_precision")
                ),
                "val_weighted_recall": _metric_value(
                    metrics.get("val_weighted_recall")
                ),
            }
        )


def _metric_value(metric) -> float:
    if metric is None:
        return 0.0

    if hasattr(metric, "detach"):
        return float(metric.detach().cpu().item())

    return float(metric)


def get_git_commit_id() -> str:
    return get_git_metadata()["commit_id"]


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


def run_dvc_command(command: list[str]) -> str:
    return run_command([sys.executable, "-m", "dvc", *command])


def get_git_metadata() -> dict[str, str]:
    status = run_command(["git", "status", "--porcelain"])

    return {
        "commit_id": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "branch", "--show-current"]),
        "is_dirty": str(bool(status)),
    }


def get_file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return "missing"

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_dvc_dataset_metadata(dvc_file: str | Path = "data.dvc") -> dict[str, str]:
    dvc_file = Path(dvc_file)
    if not dvc_file.is_absolute():
        dvc_file = PROJECT_ROOT / dvc_file

    if not dvc_file.exists():
        return {
            "dvc_data_path": "missing",
            "dvc_data_md5": "missing",
            "dvc_data_size": "missing",
            "dvc_data_nfiles": "missing",
            "dvc_remote": run_dvc_command(["remote", "default"]),
        }

    with dvc_file.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    output = data["outs"][0]
    remote = run_dvc_command(["remote", "default"])

    if remote == "unknown":
        remote = "data-storage"

    remote_url = run_dvc_command(["config", f"remote.{remote}.url"])

    return {
        "dvc_data_path": str(output.get("path", "")),
        "dvc_data_md5": str(output.get("md5", "")),
        "dvc_data_hash": str(output.get("hash", "")),
        "dvc_data_size": str(output.get("size", "")),
        "dvc_data_nfiles": str(output.get("nfiles", "")),
        "dvc_remote": remote,
        "dvc_remote_url": remote_url,
    }


def get_docker_metadata() -> dict[str, str]:
    image_name = os.getenv(
        "LANDSCAPE_API_IMAGE",
        "landscape-classifier-mlops-api",
    )
    image_id = run_command(
        [
            "docker",
            "image",
            "inspect",
            image_name,
            "--format",
            "{{.Id}}",
        ]
    )

    return {
        "docker_image_name": image_name,
        "docker_image_id": image_id,
        "dockerfile_api_sha256": get_file_sha256("Dockerfile.api"),
        "docker_compose_sha256": get_file_sha256("docker-compose.yml"),
    }


def flatten_config(cfg: DictConfig) -> dict[str, str | int | float | bool]:
    container = OmegaConf.to_container(cfg, resolve=True)
    flattened: dict[str, str | int | float | bool] = {}

    def visit(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, child_value in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, child_value)
            return

        if isinstance(value, list):
            flattened[prefix] = ",".join(str(item) for item in value)
            return

        if value is None:
            flattened[prefix] = "null"
            return

        flattened[prefix] = value

    visit("", container)
    return flattened


def save_run_metadata(
    output_path: Path,
    cfg: DictConfig,
    git_metadata: dict[str, str],
    dvc_metadata: dict[str, str],
    docker_metadata: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git": git_metadata,
        "dvc": dvc_metadata,
        "docker": docker_metadata,
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def log_mlflow_metadata(
    cfg: DictConfig,
    git_metadata: dict[str, str],
    dvc_metadata: dict[str, str],
    docker_metadata: dict[str, str],
) -> None:
    mlflow.set_tags(
        {
            "git.commit": git_metadata["commit_id"],
            "git.branch": git_metadata["branch"],
            "git.is_dirty": git_metadata["is_dirty"],
            "docker.image": docker_metadata["docker_image_name"],
            "dvc.data_md5": dvc_metadata["dvc_data_md5"],
            "model.name": cfg.model.name,
            "model.registry_name": cfg.model.registry_name,
        }
    )
    mlflow.log_params(
        {
            **{f"cfg.{key}": value for key, value in flatten_config(cfg).items()},
            **git_metadata,
            **dvc_metadata,
            **docker_metadata,
        }
    )


def set_model_alias(
    client: MlflowClient,
    registered_model_name: str,
    production_alias: str,
    run_id: str,
) -> None:
    model_versions = client.search_model_versions(f"name='{registered_model_name}'")
    current_run_versions = [
        model_version
        for model_version in model_versions
        if model_version.run_id == run_id
    ]

    if not current_run_versions:
        logger.warning(
            "Could not determine registered model version for "
            f"run_id={run_id}; alias '{production_alias}' was not updated."
        )
        return

    registered_model_version = max(
        current_run_versions,
        key=lambda model_version: int(model_version.version),
    ).version

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=production_alias,
        version=registered_model_version,
    )

    logger.info(
        f"Alias '{production_alias}' now points to "
        f"{registered_model_name} version {registered_model_version}"
    )


def build_trainer(
    cfg: DictConfig,
    mlflow_logger: MLFlowLogger,
    checkpoint_callback: ModelCheckpoint,
    history_callback: MetricHistoryCallback,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        deterministic=cfg.training.deterministic,
        logger=mlflow_logger,
        callbacks=[
            checkpoint_callback,
            history_callback,
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_checkpointing=True,
    )


@hydra.main(
    version_base=None,
    config_path=str(get_project_root() / "configs"),
    config_name="config",
)
def train(cfg: DictConfig) -> None:
    logger.info("Starting Lightning training pipeline")
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    datamodule = build_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    class_names = datamodule.class_names
    num_classes = len(class_names)

    logger.info(f"Classes found: {class_names}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"Validation samples: {len(datamodule.val_dataset)}")
    logger.info(f"Test samples: {len(datamodule.test_dataset)}")

    model = LandscapeClassifierModule(
        num_classes=num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    model_path = Path(cfg.model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_name = f"{cfg.model.name}_{timestamp}"
    git_metadata = get_git_metadata()
    dvc_metadata = get_dvc_dataset_metadata()
    docker_metadata = get_docker_metadata()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=run_name,
    )
    mlflow_logger.log_hyperparams(
        {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "optimizer": cfg.training.optimizer,
            "loss": cfg.training.loss,
            "model": cfg.model.name,
            "architecture": cfg.model.architecture,
            "num_classes": num_classes,
            "image_size": cfg.model.image_size,
            "val_size": cfg.data.val_size,
            "num_workers": cfg.data.num_workers,
            "seed": cfg.seed,
            "git_commit_id": git_metadata["commit_id"],
            "git_branch": git_metadata["branch"],
            "git_is_dirty": git_metadata["is_dirty"],
            "dvc_data_md5": dvc_metadata["dvc_data_md5"],
            "dvc_data_nfiles": dvc_metadata["dvc_data_nfiles"],
            "dvc_remote": dvc_metadata["dvc_remote"],
            "docker_image_name": docker_metadata["docker_image_name"],
            "docker_image_id": docker_metadata["docker_image_id"],
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_weighted_f1:.4f}",
        monitor=cfg.training.checkpoint_monitor,
        mode=cfg.training.checkpoint_mode,
        save_top_k=1,
    )
    history_callback = MetricHistoryCallback()
    trainer = build_trainer(
        cfg=cfg,
        mlflow_logger=mlflow_logger,
        checkpoint_callback=checkpoint_callback,
        history_callback=history_callback,
    )

    logger.info("Starting Lightning trainer.fit")
    trainer.fit(
        model,
        datamodule=datamodule,
    )

    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path:
        logger.info(f"Loading best checkpoint: {best_checkpoint_path}")
        best_model = LandscapeClassifierModule.load_from_checkpoint(
            best_checkpoint_path,
        )
    else:
        logger.warning("No best checkpoint was produced; using current model weights")
        best_model = model

    torch.save(
        best_model.model.state_dict(),
        model_path,
    )
    logger.info(f"Best model state_dict saved: {model_path}")

    trainer.test(
        best_model,
        datamodule=datamodule,
    )

    device = next(best_model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(
        best_model,
        datamodule.test_dataloader(),
        criterion,
        device,
    )

    logger.info(
        f"Test metrics: "
        f"loss={test_metrics['loss']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"precision={test_metrics['weighted_precision']:.4f}, "
        f"recall={test_metrics['weighted_recall']:.4f}, "
        f"auc={test_metrics['auc']:.4f}"
    )

    artifacts_dir = Path("artifacts")
    plots_dir = Path("plots")

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(
        cfg,
        artifacts_dir / "resolved_config.yaml",
    )
    save_run_metadata(
        artifacts_dir / "metadata" / "run_metadata.json",
        cfg=cfg,
        git_metadata=git_metadata,
        dvc_metadata=dvc_metadata,
        docker_metadata=docker_metadata,
    )

    sample_report_dir = artifacts_dir / "sample_report"
    model_artifact_dir = artifacts_dir / "model"
    model_tester_dir = artifacts_dir / "model_tester"

    save_dataset_stats(
        sample_report_dir / "dataset_stats.json",
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        class_names,
        cfg.model.image_size,
    )

    save_class_distribution(
        datamodule.train_dataset,
        class_names,
        sample_report_dir / "train",
    )
    save_class_distribution(
        datamodule.val_dataset,
        class_names,
        sample_report_dir / "val",
    )
    save_class_distribution(
        datamodule.test_dataset,
        class_names,
        sample_report_dir / "test",
    )

    save_class_balance(
        datamodule.train_dataset,
        class_names,
        sample_report_dir / "train" / "class_balance.csv",
    )
    save_class_balance(
        datamodule.val_dataset,
        class_names,
        sample_report_dir / "val" / "class_balance.csv",
    )
    save_class_balance(
        datamodule.test_dataset,
        class_names,
        sample_report_dir / "test" / "class_balance.csv",
    )

    save_sample_images(
        datamodule.train_dataloader(),
        class_names,
        sample_report_dir / "train" / "sample_images.png",
    )
    save_sample_images(
        datamodule.val_dataloader(),
        class_names,
        sample_report_dir / "val" / "sample_images.png",
    )
    save_sample_images(
        datamodule.test_dataloader(),
        class_names,
        sample_report_dir / "test" / "sample_images.png",
    )

    report = get_classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        class_names,
    )
    confusion_matrix_array = get_confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
    )

    save_model_tester_artifacts(
        model_tester_dir,
        test_metrics["y_true"],
        test_metrics["y_pred"],
        class_names,
        report,
        confusion_matrix_array,
    )
    save_misclassified_examples(
        best_model,
        datamodule.test_dataloader(),
        class_names,
        device,
        model_tester_dir / "misclassified_examples.png",
    )

    training_curves_path = model_tester_dir / "training_curves.png"
    save_training_curves(
        history_callback.history,
        training_curves_path,
    )
    shutil.copy(
        training_curves_path,
        plots_dir / "training_curves.png",
    )

    save_model_info(
        model_artifact_dir,
        model_name=model_path.name,
        class_names=class_names,
        image_size=cfg.model.image_size,
        metrics={
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_precision": test_metrics["weighted_precision"],
            "test_weighted_recall": test_metrics["weighted_recall"],
            "test_auc": test_metrics["auc"],
        },
    )
    save_model_summary(
        best_model,
        model_artifact_dir / "model_summary.txt",
        cfg.model.image_size,
    )
    shutil.copy(
        model_path,
        model_artifact_dir / model_path.name,
    )

    if Path("logs/train.log").exists():
        shutil.copy(
            "logs/train.log",
            artifacts_dir / "train.log",
        )

    run_id = mlflow_logger.run_id
    with mlflow.start_run(run_id=run_id):
        log_mlflow_metadata(
            cfg=cfg,
            git_metadata=git_metadata,
            dvc_metadata=dvc_metadata,
            docker_metadata=docker_metadata,
        )
        mlflow.log_metrics(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_precision": test_metrics["weighted_precision"],
                "test_weighted_recall": test_metrics["weighted_recall"],
                "test_auc": test_metrics["auc"],
            }
        )
        mlflow.log_artifacts(
            local_dir=str(artifacts_dir),
            artifact_path="",
        )
        mlflow.log_artifacts(
            local_dir=str(plots_dir),
            artifact_path="plots",
        )
        if best_checkpoint_path:
            mlflow.log_artifact(
                local_path=best_checkpoint_path,
                artifact_path="checkpoints",
            )
        mlflow.log_artifact(
            local_path=str(model_path),
            artifact_path="model_state_dict",
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_model.model,
            artifact_path="registered_model",
            registered_model_name=cfg.model.registry_name,
        )

    set_model_alias(
        client=MlflowClient(),
        registered_model_name=cfg.model.registry_name,
        production_alias=cfg.inference.production_alias,
        run_id=run_id,
    )

    logger.info("Training pipeline finished successfully")


if __name__ == "__main__":
    train()
