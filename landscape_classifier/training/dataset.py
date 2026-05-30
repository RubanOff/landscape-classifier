from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from landscape_classifier.dvc_utils import ensure_dvc_paths


def build_train_transform(image_size: int):
    """Build training image transforms."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(image_size: int):
    """Build evaluation image transforms."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def split_indices(dataset_size: int, val_size: float, seed: int):
    """Split dataset indices into train and val."""
    val_count = int(dataset_size * val_size)
    train_count = dataset_size - val_count

    indices = torch.randperm(
        dataset_size,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    return indices[:train_count], indices[train_count:]


class LandscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str | Path,
        test_dir: str | Path,
        batch_size: int,
        image_size: int,
        val_size: float,
        seed: int,
        num_workers: int,
        pull_with_dvc: bool = True,
    ):
        """Store datamodule settings."""
        super().__init__()
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_size = val_size
        self.seed = seed
        self.num_workers = num_workers
        self.pull_with_dvc = pull_with_dvc
        self.class_names: list[str] = []

    def prepare_data(self) -> None:
        """Pull data before setup."""
        if self.pull_with_dvc:
            ensure_dvc_paths([self.train_dir, self.test_dir])

    def setup(self, stage: str | None = None) -> None:
        """Create train, val, and test datasets."""
        train_source = datasets.ImageFolder(
            self.train_dir,
            transform=build_train_transform(self.image_size),
        )
        val_source = datasets.ImageFolder(
            self.train_dir,
            transform=build_eval_transform(self.image_size),
        )

        train_indices, val_indices = split_indices(
            len(train_source),
            self.val_size,
            self.seed,
        )

        self.train_dataset = Subset(train_source, train_indices)
        self.val_dataset = Subset(val_source, val_indices)
        self.test_dataset = datasets.ImageFolder(
            self.test_dir,
            transform=build_eval_transform(self.image_size),
        )
        self.class_names = train_source.classes

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )


def build_datamodule(cfg) -> LandscapeDataModule:
    """Build datamodule from config."""
    return LandscapeDataModule(
        train_dir=cfg.data.train_dir,
        test_dir=cfg.data.test_dir,
        batch_size=cfg.training.batch_size,
        image_size=cfg.model.image_size,
        val_size=cfg.data.val_size,
        seed=cfg.seed,
        num_workers=cfg.data.num_workers,
    )


def get_dataloaders(
    train_dir,
    test_dir,
    batch_size: int,
    image_size: int,
    val_size: float,
    seed: int,
    num_workers: int,
):
    """Build dataloaders directly."""
    datamodule = LandscapeDataModule(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        image_size=image_size,
        val_size=val_size,
        seed=seed,
        num_workers=num_workers,
        pull_with_dvc=False,
    )
    datamodule.setup()

    return (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        datamodule.class_names,
    )
