"""Load and validate CSV datasets produced by the landmark collector."""
from __future__ import annotations

import re
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd

_FEATURE_COLUMN = re.compile(r"^f\d+$")


@overload
def load_landmark_csv(path: str | Path, label_column: str = "label", *, return_split: bool = False) -> tuple[pd.DataFrame, pd.Series]: ...


@overload
def load_landmark_csv(path: str | Path, label_column: str = "label", *, return_split: bool) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]: ...


def load_landmark_csv(
    path: str | Path,
    label_column: str = "label",
    *,
    return_split: bool = False,
):
    """Return features and labels, optionally with validated predefined split metadata.

    Split metadata is never included in the model feature matrix. A dataset may
    have no split column (or a completely blank one), in which case callers can
    use a reproducible stratified random split instead.
    """
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
    columns = sorted(
        (c for c in frame.columns if _FEATURE_COLUMN.fullmatch(str(c))),
        key=lambda c: int(str(c)[1:]),
    )
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

    result = (features.astype(np.float32), labels.astype(str).rename(label_column))
    if not return_split:
        return result

    if "split" not in frame.columns:
        return (*result, None)
    raw_split = frame["split"].astype("string").str.strip().str.lower()
    if raw_split.isna().all() or raw_split.fillna("").eq("").all():
        return (*result, None)
    if raw_split.isna().any() or raw_split.eq("").any():
        raise ValueError("Split metadata must be populated for every row or left entirely blank")
    allowed = {"train", "test"}
    unexpected = sorted(set(raw_split.astype(str)) - allowed)
    if unexpected:
        raise ValueError("Split values must be 'train' or 'test'; got: " + ", ".join(unexpected))
    splits = raw_split.astype(str).rename("split")
    if set(splits) != allowed:
        raise ValueError("Predefined split metadata must contain both train and test rows")
    return (*result, splits)
