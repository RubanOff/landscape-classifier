import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from landscape_classifier.training.model import get_model


class LandscapeClassifierModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        pretrained: bool = True,
    ):
        """Initialize the Lightning classifier."""
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="weighted",
        )
        self.train_weighted_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="weighted",
        )
        self.train_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="weighted",
        )
        self.train_recall = MulticlassRecall(
            num_classes=num_classes,
            average="weighted",
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="weighted",
        )
        self.test_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="weighted",
        )
        self.val_weighted_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="weighted",
        )
        self.val_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="weighted",
        )
        self.val_recall = MulticlassRecall(
            num_classes=num_classes,
            average="weighted",
        )
        self.test_weighted_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="weighted",
        )
        self.test_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="weighted",
        )
        self.test_recall = MulticlassRecall(
            num_classes=num_classes,
            average="weighted",
        )

    def forward(self, images):
        """Run the model forward pass."""
        return self.model(images)

    def training_step(self, batch, batch_idx):
        """Run one training batch."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        self.train_accuracy.update(predictions, targets)
        self.train_weighted_f1.update(predictions, targets)
        self.train_precision.update(predictions, targets)
        self.train_recall.update(predictions, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_weighted_f1",
            self.train_weighted_f1,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_weighted_precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_weighted_recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Run one validation batch."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        self.val_accuracy.update(predictions, targets)
        self.val_weighted_f1.update(predictions, targets)
        self.val_precision.update(predictions, targets)
        self.val_recall.update(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_weighted_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_weighted_recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_weighted_f1",
            self.val_weighted_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        """Run one test batch."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        self.test_accuracy.update(predictions, targets)
        self.test_weighted_f1.update(predictions, targets)
        self.test_precision.update(predictions, targets)
        self.test_recall.update(predictions, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log(
            "test_weighted_f1",
            self.test_weighted_f1,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_weighted_precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_weighted_recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        """Create the optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
