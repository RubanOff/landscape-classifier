from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from landscape_classifier.config import load_config as load_hydra_config


def load_config() -> DictConfig:
    """Load the training config."""
    return load_hydra_config()


def save_resolved_config(cfg: DictConfig, output_path: Path) -> None:
    """Save the resolved Hydra config."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))
