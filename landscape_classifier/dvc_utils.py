import subprocess
from pathlib import Path

from landscape_classifier.training.logger import get_logger

logger = get_logger("landscape-dvc")


def ensure_dvc_paths(paths: list[str | Path]) -> None:
    """Pull missing DVC-managed paths."""
    missing_paths = [Path(path) for path in paths if not Path(path).exists()]

    if not missing_paths:
        return

    logger.info(
        "Missing DVC-managed paths: "
        + ", ".join(str(path) for path in missing_paths)
        + ". Running dvc pull."
    )

    subprocess.run(
        ["dvc", "pull"],
        check=True,
    )
