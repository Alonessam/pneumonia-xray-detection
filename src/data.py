from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def discover_class_names(data_dir: str | Path) -> list[str]:
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train klasörü bulunamadı: {train_dir}. Beklenen yapı: data/chest_xray/train/NORMAL ve PNEUMONIA"
        )

    available = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    preferred = [name for name in CLASS_NAMES if name in available]
    if len(preferred) == len(CLASS_NAMES):
        return preferred
    if len(available) < 2:
        raise ValueError(f"En az iki sınıf klasörü bekleniyor. Bulunanlar: {available}")
    return available


def collect_samples(
    data_dir: str | Path,
    split_names: list[str],
    class_names: list[str],
) -> list[tuple[Path, int]]:
    data_dir = Path(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []

    for split in split_names:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for class_name, label in class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((image_path, label))

    if not samples:
        raise FileNotFoundError(f"{data_dir} içinde {split_names} için görüntü bulunamadı.")
    return samples


def stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    for label_samples in grouped.values():
        rng.shuffle(label_samples)
        n_val = max(1, int(len(label_samples) * val_ratio)) if len(label_samples) > 1 else 0
        val_samples.extend(label_samples[:n_val])
        train_samples.extend(label_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    steps: list[transforms.transforms.Transform] = [transforms.Resize((img_size, img_size))]
    if train:
        steps = [transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0))]
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def class_counts(samples: list[tuple[Path, int]], class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for _, label in samples:
        counts[class_names[label]] += 1
    return counts


def make_dataloaders(
    data_dir: str | Path,
    img_size: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
    include_kaggle_val_in_train_pool: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    class_names = discover_class_names(data_dir)
    train_splits = ["train", "val"] if include_kaggle_val_in_train_pool else ["train"]
    train_pool = collect_samples(data_dir, train_splits, class_names)
    train_samples, val_samples = stratified_split(train_pool, val_ratio=val_ratio, seed=seed)
    test_samples = collect_samples(data_dir, ["test"], class_names)

    train_dataset = ChestXrayDataset(train_samples, class_names, transform=build_transforms(img_size, train=True))
    val_dataset = ChestXrayDataset(val_samples, class_names, transform=build_transforms(img_size, train=False))
    test_dataset = ChestXrayDataset(test_samples, class_names, transform=build_transforms(img_size, train=False))

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    info = {
        "class_names": class_names,
        "train_counts": class_counts(train_samples, class_names),
        "val_counts": class_counts(val_samples, class_names),
        "test_counts": class_counts(test_samples, class_names),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
    }
    return loaders, info


def make_class_weights(train_counts: dict[str, int], class_names: list[str]) -> torch.Tensor:
    counts = torch.tensor([train_counts[name] for name in class_names], dtype=torch.float32)
    total = counts.sum()
    weights = total / (len(class_names) * counts.clamp_min(1))
    return weights
