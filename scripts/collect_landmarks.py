"""Extract hand landmarks from labeled image folders into a CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.preprocessing.landmarks import flatten_landmarks
from src.vision.hand_detector import HandDetector
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_COLUMNS = [f"f{i}" for i in range(63)]
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["label", "handedness", "hand_confidence"]

def collect(input_dir: Path, output_csv: Path) -> int:
    """Extract one sample per detected hand and return the number of samples."""
    input_dir, output_csv = Path(input_dir), Path(output_csv)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found under {input_dir}")
    rows = []
    with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
        for image_path in images:
            label = image_path.parent.name.strip()
            image = cv2.imread(str(image_path))
            if image is None or not label:
                continue
            for hand in detector.detect(image):
                row = {name: float(value) for name, value in zip(FEATURE_COLUMNS, flatten_landmarks(hand.landmarks))}
                row.update(label=label, handedness=hand.handedness or "", hand_confidence=float(hand.confidence) if hand.confidence is not None else np.nan)
                rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
    if not rows:
        raise ValueError("No hands were detected in the supplied images; no usable dataset was produced")
    return len(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 21 hand landmarks from labeled images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing one subfolder per sign label")
    parser.add_argument("output_csv", type=Path, help="Destination CSV, e.g. data/processed/landmarks.csv")
    args = parser.parse_args()
    try:
        count = collect(args.input_dir, args.output_csv)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {count} detected-hand samples to {args.output_csv}")

if __name__ == "__main__":
    main()
