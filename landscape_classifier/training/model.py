import torch.nn as nn
from torchvision import models


def get_model(num_classes: int, pretrained: bool = True):
    """Build a ResNet18 classifier."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.resnet18(
        weights=weights,
    )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
