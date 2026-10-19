"""Extract hand landmarks from labeled image folders into a CSV."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.landmarks import flatten_landmarks
from src.vision.hand_detector import HandDetector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_COLUMNS = [f"f{i}" for i in range(63)]
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["label", "handedness", "hand_confidence", "split"]
_SPLIT_NAMES = {
    "train": "train", "training": "train",
    "test": "test", "testing": "test",
}


def image_records(input_dir: Path) -> list[tuple[Path, str, str]]:
    """Return (path, label, split) records; split is empty when not predefined.

    Recognizes either root/<LABEL>/image or root/{train,test}/<LABEL>/image.
    """
    records = []
    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(input_dir).parts
        if len(relative) >= 3 and relative[0].lower() in _SPLIT_NAMES:
            split = _SPLIT_NAMES[relative[0].lower()]
            label = relative[1].strip()
        else:
            split = ""
            label = path.parent.name.strip()
        if label:
            records.append((path, label, split))
    return records


def select_image_records(
    records: list[tuple[Path, str, str]],
    max_images_per_class: int | None = None,
    random_state: int = 42,
) -> list[tuple[Path, str, str]]:
    """Select a reproducible maximum per label and predefined split."""
    if max_images_per_class is not None and max_images_per_class < 1:
        raise ValueError("max_images_per_class must be positive")
    if max_images_per_class is None:
        return sorted(records, key=lambda item: str(item[0]))
    grouped: dict[tuple[str, str], list[tuple[Path, str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record[1], record[2])].append(record)
    rng = random.Random(random_state)
    selected = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: str(item[0]))
        if len(group) > max_images_per_class:
            group = rng.sample(group, max_images_per_class)
        selected.extend(group)
    return sorted(selected, key=lambda item: str(item[0]))


def collect(
    input_dir: Path,
    output_csv: Path,
    *,
    max_images_per_class: int | None = None,
    random_state: int = 42,
) -> int:
    """Extract one sample per detected hand and return the number of samples."""
    input_dir, output_csv = Path(input_dir), Path(output_csv)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")
    records = image_records(input_dir)
    if not records:
        raise ValueError(f"No supported images found under {input_dir}")
    records = select_image_records(records, max_images_per_class, random_state)
    rows = []
    processed_images = 0
    with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
        for image_path, label, split in records:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            processed_images += 1
            for hand in detector.detect(image):
                row = {
                    name: float(value)
                    for name, value in zip(FEATURE_COLUMNS, flatten_landmarks(hand.landmarks))
                }
                row.update(
                    label=label,
                    handedness=hand.handedness or "",
                    hand_confidence=float(hand.confidence) if hand.confidence is not None else np.nan,
                    split=split,
                )
                rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
    if not rows:
        raise ValueError("No hands were detected in the supplied images; no usable dataset was produced")
    print(f"Images processed: {processed_images}/{len(records)}")
    print(f"Detected hands by label: {pd.Series([row['label'] for row in rows]).value_counts().sort_index().to_dict()}")
    if any(row["split"] for row in rows):
        print(f"Detected hands by split: {pd.Series([row['split'] for row in rows]).value_counts().to_dict()}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 21 hand landmarks from labeled images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing label folders or train/test label folders")
    parser.add_argument("output_csv", type=Path, help="Destination CSV, e.g. data/processed/landmarks.csv")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Optional deterministic cap per label and split")
    parser.add_argument("--random-state", type=int, default=42, help="Seed used when sampling images")
    args = parser.parse_args()
    try:
        count = collect(
            args.input_dir,
            args.output_csv,
            max_images_per_class=args.max_images_per_class,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {count} detected-hand samples to {args.output_csv}")


if __name__ == "__main__":
    main()
