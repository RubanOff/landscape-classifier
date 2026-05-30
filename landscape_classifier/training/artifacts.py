import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchinfo import summary


def get_pyplot():
    """Load pyplot in headless mode."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def save_class_distribution(dataset, class_names, output_dir):
    """Save class distribution artifacts."""
    plt = get_pyplot()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        targets = [dataset.dataset.targets[idx] for idx in dataset.indices]

    rows = []

    for class_idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class_id": class_idx,
                "class_name": class_name,
                "count": targets.count(class_idx),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / "class_distribution.csv",
        index=False,
    )

    plt.figure(figsize=(8, 4))

    bars = plt.bar(
        df["class_name"],
        df["count"],
    )

    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    for bar in bars:
        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 5,
            int(yval),
            ha="center",
        )

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        output_dir / "class_distribution.png",
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()


def save_model_info(output_dir, model_name, class_names, image_size, metrics):
    """Save model metadata as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "model_name": model_name,
        "architecture": "ResNet18",
        "pretrained": True,
        "image_size": image_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "test_metrics": metrics,
    }

    with open(output_dir / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)


def save_model_tester_artifacts(
    output_dir,
    y_true,
    y_pred,
    class_names,
    classification_report_dict,
    confusion_matrix_array,
):
    """Save evaluation reports and matrix."""
    plt = get_pyplot()

    from sklearn.metrics import ConfusionMatrixDisplay

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(classification_report_dict).transpose()

    report_df = report_df.reset_index()
    report_df = report_df.rename(columns={"index": "class_name"})

    report_df.to_csv(
        output_dir / "classification_report.csv",
        index=False,
    )

    cm_df = pd.DataFrame(
        confusion_matrix_array,
        index=class_names,
        columns=class_names,
    )

    cm_df = cm_df.reset_index()
    cm_df = cm_df.rename(columns={"index": "true_class"})

    cm_df.to_csv(
        output_dir / "confusion_matrix.csv",
        index=False,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_array,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    disp.plot(
        ax=ax,
        xticks_rotation=30,
        cmap="Blues",
        colorbar=False,
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()


def denormalize_image(image):
    """Convert normalized tensor to image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image.permute(1, 2, 0).cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def save_sample_images(loader, class_names, output_path, n_images=9):
    """Save a grid of sample images."""
    plt = get_pyplot()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))

    n_images = min(n_images, len(images))

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(n_images):
        image = denormalize_image(images[i])
        label = labels[i].item()

        axes[i].imshow(image)
        axes[i].set_title(class_names[label], fontsize=9)
        axes[i].axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_dataset_stats(
    output_path,
    train_loader,
    val_loader,
    test_loader,
    class_names,
    image_size,
):
    """Save dataset summary stats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": image_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4, ensure_ascii=False)


def save_class_balance(dataset, class_names, output_path):
    """Save class balance as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        targets = [dataset.dataset.targets[idx] for idx in dataset.indices]

    total = len(targets)

    rows = []

    for class_idx, class_name in enumerate(class_names):
        count = targets.count(class_idx)

        rows.append(
            {
                "class_id": class_idx,
                "class_name": class_name,
                "count": count,
                "percentage": round(count / total * 100, 2),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def save_training_curves(history, output_path):
    """Save training metric curves."""
    plt = get_pyplot()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["val_weighted_f1"], label="val_weighted_f1")
    axes[2].set_title("Validation Weighted F1")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    df.to_csv(output_path.parent / "training_history.csv", index=False)


def save_model_summary(model, output_path, image_size):
    """Save a text model summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_summary = summary(
        model,
        input_size=(1, 3, image_size, image_size),
        verbose=0,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(str(model_summary))


@torch.no_grad()
def save_misclassified_examples(
    model,
    loader,
    class_names,
    device,
    output_path,
    max_images=9,
):
    """Save misclassified test examples."""
    plt = get_pyplot()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    examples = []

    for images, targets in loader:
        images_device = images.to(device)
        outputs = model(images_device)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        for image, target, pred, prob in zip(
            images,
            targets,
            preds.cpu(),
            probs.cpu(),
            strict=True,
        ):
            if target.item() != pred.item():
                examples.append(
                    {
                        "image": image,
                        "true": target.item(),
                        "pred": pred.item(),
                        "confidence": prob[pred].item(),
                    }
                )

            if len(examples) >= max_images:
                break

        if len(examples) >= max_images:
            break

    if not examples:
        return

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    axes = axes.flatten()

    for i, example in enumerate(examples):
        image = denormalize_image(example["image"])

        true_name = class_names[example["true"]]
        pred_name = class_names[example["pred"]]
        confidence = example["confidence"]

        axes[i].imshow(image)
        axes[i].set_title(
            f"true: {true_name}\npred: {pred_name}\nconf: {confidence:.2f}",
            fontsize=8,
        )
        axes[i].axis("off")

    for i in range(len(examples), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
