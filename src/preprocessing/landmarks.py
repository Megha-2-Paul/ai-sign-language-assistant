"""Feature engineering utilities for hand landmark coordinates."""
from __future__ import annotations
import numpy as np
EXPECTED_LANDMARKS = 21

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Center 21 landmarks on the wrist and scale by hand size."""
    try:
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmarks must be numeric") from exc
    if points.ndim != 2 or points.shape[0] != EXPECTED_LANDMARKS or points.shape[1] not in (2, 3):
        raise ValueError("Landmarks must have shape (21, 2) or (21, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks must contain only finite values")
    centered = points - points[0]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 1e-8:
        return np.zeros_like(centered)
    return centered / scale

def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Convert normalized landmarks to a one-dimensional feature vector."""
    return normalize_landmarks(landmarks).reshape(-1)
