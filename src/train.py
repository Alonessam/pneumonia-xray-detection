from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from .config import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DEFAULT_REPORTS_DIR
from .data import make_class_weights, make_dataloaders
from .models import build_model
from .utils import count_parameters, ensure_dir, get_device, load_checkpoint, save_checkpoint, save_json, set_seed


def run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pneumonia X-ray classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", choices=["simple_cnn", "resnet50"], default="simple_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--resume", type=Path, default=None, help="Fine-tuning için önceki checkpoint yolu.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.reports_dir)

    loaders, data_info = make_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    class_names = data_info["class_names"]
    device = get_device()

    model = build_model(
        args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        if checkpoint["model_name"] != args.model:
            raise ValueError(f"Checkpoint modeli {checkpoint['model_name']}, istenen model {args.model}.")
        model.load_state_dict(checkpoint["model_state"])
        print(f"Resumed from checkpoint: {args.resume}")

    total_params, trainable_params = count_parameters(model)
    class_weights = None
    if not args.no_class_weights:
        class_weights = make_class_weights(data_info["train_counts"], class_names).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Classes: {class_names}")
    print(f"Data info: {data_info}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    best_val_loss = float("inf")
    suffix = ""
    if args.model == "resnet50":
        suffix = "_frozen" if args.freeze_backbone else "_finetuned"

    best_path = args.out_dir / f"best_{args.model}{suffix}.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, loaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model=model,
                model_name=args.model,
                class_names=class_names,
                img_size=args.img_size,
                metrics={"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch},
                extra={
                    "data_info": data_info,
                    "freeze_backbone": args.freeze_backbone,
                    "pretrained": not args.no_pretrained,
                },
            )
            print(f"Saved best checkpoint: {best_path}")

    history_path = args.reports_dir / f"history_{args.model}{suffix}.csv"
    write_history(history_path, history)
    save_json({"data_info": data_info, "best_val_loss": best_val_loss}, args.reports_dir / f"train_info_{args.model}{suffix}.json")
    print(f"Training finished. History: {history_path}")


if __name__ == "__main__":
    main()
