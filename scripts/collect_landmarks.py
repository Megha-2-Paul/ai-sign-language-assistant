"""Extract hand landmarks from an image folder into a CSV dataset.

Expected input layout:
    input_dir/<label>/*.jpg

Output contains one row per detected hand, landmark coordinates, and label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.vision.hand_detector import HandDetector
from src.preprocessing.landmarks import flatten_landmarks


def collect(input_dir: Path, output_csv: Path) -> int:
    rows = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with HandDetector(static_image_mode=True) as detector:
        for image_path in sorted(input_dir.rglob("*")):
            if image_path.suffix.lower() not in extensions:
                continue
            label = image_path.parent.name
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            for hand in detector.detect(image):
                features = flatten_landmarks(hand.landmarks)
                row = {f"f{i}": float(value) for i, value in enumerate(features)}
                row["label"] = label
                row["handedness"] = hand.handedness
                row["hand_confidence"] = hand.confidence
                rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hand landmarks from labeled images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    count = collect(args.input_dir, args.output_csv)
    print(f"Wrote {count} detected-hand samples to {args.output_csv}")


if __name__ == "__main__":
    main()
