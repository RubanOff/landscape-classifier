from pathlib import Path

from PIL import Image

from landscape_classifier.inference.main import health
from landscape_classifier.inference.utils import build_predict_transform
from landscape_classifier.training.dataset import LandscapeDataModule


def create_image_dataset(root: Path, classes: list[str], images_per_class: int = 2):
    """Create a tiny image dataset."""
    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_idx in range(images_per_class):
            image = Image.new(
                "RGB",
                (16, 16),
                color=(image_idx * 20, 80, 120),
            )
            image.save(class_dir / f"{image_idx}.jpg")


def test_datamodule_uses_configurable_split_and_batch_size(tmp_path):
    """Check datamodule split and batch size."""
    classes = ["forest", "sea"]
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    create_image_dataset(train_dir, classes, images_per_class=3)
    create_image_dataset(test_dir, classes, images_per_class=1)

    datamodule = LandscapeDataModule(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=2,
        image_size=32,
        val_size=0.5,
        seed=42,
        num_workers=0,
        pull_with_dvc=False,
    )
    datamodule.setup()

    assert len(datamodule.train_dataset) == 3
    assert len(datamodule.val_dataset) == 3
    assert len(datamodule.test_dataset) == 2
    assert datamodule.train_dataloader().batch_size == 2
    assert datamodule.class_names == classes


def test_predict_transform_uses_configured_image_size():
    """Check prediction image resizing."""
    transform = build_predict_transform(32)
    image = Image.new("RGB", (16, 16), color=(10, 20, 30))

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 32, 32)


def test_health_does_not_load_model():
    """Check health endpoint laziness."""
    assert health() == {
        "status": "ok",
        "model_loaded": False,
    }
