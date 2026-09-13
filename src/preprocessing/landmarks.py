"""Feature engineering utilities for hand landmark coordinates."""

from __future__ import annotations

import numpy as np


EXPECTED_LANDMARKS = 21


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize 21 hand landmarks relative to the wrist and hand scale.

    Input shape must be (21, 2) or (21, 3). The wrist becomes the origin and
    the coordinates are divided by the maximum distance from the wrist.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape[0] != EXPECTED_LANDMARKS:
        raise ValueError(f"Expected {EXPECTED_LANDMARKS} landmarks, got {points.shape[0]}")
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Landmarks must have shape (21, 2) or (21, 3)")

    centered = points - points[0]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 1e-8:
        return np.zeros_like(centered)
    return centered / scale


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Convert normalized landmarks to a one-dimensional model feature vector."""
    normalized = normalize_landmarks(landmarks)
    return normalized.reshape(-1)
