import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(all_targets, all_preds)

    weighted_f1 = f1_score(
        all_targets,
        all_preds,
        average="weighted",
    )

    macro_f1 = f1_score(
        all_targets,
        all_preds,
        average="macro",
    )

    weighted_precision = precision_score(
        all_targets,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        all_targets,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    auc = roc_auc_score(
        np.array(all_targets),
        np.array(all_probs),
        multi_class="ovr",
        average="weighted",
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "auc": auc,
        "y_true": np.array(all_targets),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }


def get_classification_report(y_true, y_pred, class_names):
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
