import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
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
                "val_loss": _metric_value(metrics.get("val_loss")),
                "val_accuracy": _metric_value(metrics.get("val_accuracy")),
                "val_weighted_f1": _metric_value(metrics.get("val_weighted_f1")),
            }
        )


def _metric_value(metric) -> float:
    if metric is None:
        return 0.0

    if hasattr(metric, "detach"):
        return float(metric.detach().cpu().item())

    return float(metric)


def get_git_commit_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


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
            "git_commit_id": get_git_commit_id(),
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
