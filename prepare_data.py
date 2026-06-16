from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Kaggle chest X-ray zip into data/chest_xray.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path(r"C:\Users\samet\Downloads\Compressed\archive.zip"),
        help="Kaggle archive.zip path.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Destination data directory.",
    )
    return parser.parse_args()


def should_extract(name: str) -> bool:
    name = name.replace("\\", "/")
    allowed_prefixes = ("chest_xray/train/", "chest_xray/test/", "chest_xray/val/")
    if not name.startswith(allowed_prefixes):
        return False
    if "__MACOSX" in name or name.endswith(".DS_Store") or "/._" in name:
        return False
    return True


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(f"Zip bulunamadı: {args.zip}")

    args.dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(args.zip) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not should_extract(name):
                continue

            target = args.dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

            count += 1
            if count % 500 == 0:
                print(f"Extracted {count} files...")

    print(f"Done. Extracted {count} files into {args.dest / 'chest_xray'}")


if __name__ == "__main__":
    main()
