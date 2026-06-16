from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DEFAULT_REPORTS_DIR, POSITIVE_CLASS
from .data import make_dataloaders
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, save_json, set_seed


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def dataset_paths(loader: torch.utils.data.DataLoader) -> list[str]:
    dataset = loader.dataset
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []
    return [str(path) for path, _label in samples]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, float]:
    positive_idx = class_names.index(POSITIVE_CLASS) if POSITIVE_CLASS in class_names else 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=positive_idx,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true == positive_idx, y_prob[:, positive_idx]))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Gerçek Etiket",
        xlabel="Tahmin",
        title="Confusion Matrix",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pneumonia X-ray classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODELS_DIR / "best_resnet50_finetuned.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.reports_dir)
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    loaders, data_info = make_dataloaders(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)
    paths = dataset_paths(loaders["test"])
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    stem = args.checkpoint.stem
    save_json({"metrics": metrics, "data_info": data_info}, args.reports_dir / f"metrics_{stem}.json")
    save_confusion_matrix(cm, class_names, args.reports_dir / f"confusion_matrix_{stem}.png")

    rows = []
    positive_idx = class_names.index(POSITIVE_CLASS) if POSITIVE_CLASS in class_names else 1
    for idx, (true_idx, pred_idx, probs) in enumerate(zip(y_true, y_pred, y_prob)):
        rows.append(
            {
                "image_path": paths[idx] if idx < len(paths) else "",
                "true_label": class_names[int(true_idx)],
                "pred_label": class_names[int(pred_idx)],
                "pneumonia_probability": float(probs[positive_idx]),
            }
        )
    pd.DataFrame(rows).to_csv(args.reports_dir / f"predictions_{stem}.csv", index=False)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Metrics: {metrics}")
    print(f"Confusion matrix saved to: {args.reports_dir / f'confusion_matrix_{stem}.png'}")


if __name__ == "__main__":
    main()
