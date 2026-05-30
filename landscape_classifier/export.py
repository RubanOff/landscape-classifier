from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig

from landscape_classifier.config import get_project_root, load_config
from landscape_classifier.dvc_utils import ensure_dvc_paths
from landscape_classifier.training.logger import get_logger
from landscape_classifier.training.model import get_model

logger = get_logger("landscape-export")


def export_model_to_onnx(
    model: torch.nn.Module,
    cfg: DictConfig,
    onnx_path: Path,
) -> Path:
    """Export a PyTorch model to ONNX."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        cfg.model.image_size,
        cfg.model.image_size,
    )

    dynamic_axes = None
    if cfg.export.dynamic_batch:
        dynamic_axes = {
            cfg.export.input_name: {0: "batch"},
            cfg.export.output_name: {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=cfg.export.opset_version,
        do_constant_folding=True,
        input_names=[cfg.export.input_name],
        output_names=[cfg.export.output_name],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    logger.info(f"ONNX model exported: {onnx_path}")
    return onnx_path


def export_onnx_from_config(cfg: DictConfig) -> Path:
    """Export local model weights to ONNX."""
    model_path = Path(cfg.model.path)

    ensure_dvc_paths([model_path])

    model = get_model(
        num_classes=cfg.model.num_classes,
        pretrained=False,
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        )
    )

    return export_model_to_onnx(
        model=model,
        cfg=cfg,
        onnx_path=Path(cfg.export.onnx_path),
    )


def export_production_model_to_onnx(cfg: DictConfig) -> Path:
    """Export the MLflow production model to ONNX."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model_uri = f"models:/{cfg.model.registry_name}@{cfg.inference.production_alias}"
    logger.info(f"Loading production model from MLflow Registry: {model_uri}")

    model = mlflow.pytorch.load_model(
        model_uri=model_uri,
        map_location="cpu",
    )

    return export_model_to_onnx(
        model=model,
        cfg=cfg,
        onnx_path=Path(cfg.export.onnx_path),
    )


def export_onnx_command() -> None:
    """Run ONNX export from the CLI."""
    cfg = load_config()
    export_production_model_to_onnx(cfg)


@hydra.main(
    version_base=None,
    config_path=str(get_project_root() / "configs"),
    config_name="config",
)
def export_onnx(cfg: DictConfig) -> None:
    """Run ONNX export with Hydra."""
    export_production_model_to_onnx(cfg)


if __name__ == "__main__":
    export_onnx()
