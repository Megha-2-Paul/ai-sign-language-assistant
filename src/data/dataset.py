"""Utilities for loading and validating landmark datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_landmark_csv(path: str | Path, label_column: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV containing numeric landmark features and a label column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    features = frame.drop(columns=[label_column])
    labels = frame[label_column]

    if features.empty:
        raise ValueError("Dataset contains no feature columns")
    if features.isnull().any().any():
        raise ValueError("Dataset contains missing feature values")

    return features, labels
