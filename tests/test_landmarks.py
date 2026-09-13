import numpy as np
import pytest

from src.preprocessing.landmarks import flatten_landmarks, normalize_landmarks


def test_normalize_landmarks_centers_on_wrist():
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [10, 20, 30]
    points[1] = [12, 20, 30]

    normalized = normalize_landmarks(points)

    assert normalized.shape == (21, 3)
    assert np.allclose(normalized[0], [0, 0, 0])
    assert np.allclose(normalized[1], [1, 0, 0])


def test_flatten_landmarks_has_expected_length():
    points = np.random.default_rng(42).random((21, 3), dtype=np.float32)
    assert flatten_landmarks(points).shape == (63,)


def test_invalid_landmark_count_raises():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3), dtype=np.float32))
