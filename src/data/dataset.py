"""Load and validate CSV datasets produced by the landmark collector."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
_FEATURE_COLUMN = re.compile(r"^f\d+$")

def load_landmark_csv(path: str | Path, label_column: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """Return numeric landmark features and labels, excluding capture metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Dataset contains no samples")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    columns = sorted((c for c in frame.columns if _FEATURE_COLUMN.fullmatch(str(c))), key=lambda c: int(str(c)[1:]))
    if not columns:
        raise ValueError("Dataset contains no landmark feature columns (expected f0, f1, ...)")
    if columns != [f"f{i}" for i in range(63)]:
        raise ValueError("Expected exactly 63 landmark feature columns: f0 through f62")
    labels = frame[label_column].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Dataset contains missing or empty labels")
    features = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric landmark values")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite landmark values")
    return features.astype(np.float32), labels.astype(str).rename(label_column)
