from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def get_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "configs").is_dir() and (cwd / "landscape_classifier").is_dir():
        return cwd

    return Path(__file__).resolve().parents[1]


def load_config(config_name: str = "config") -> DictConfig:
    config_dir = get_project_root() / "configs"

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        return compose(config_name=config_name)
