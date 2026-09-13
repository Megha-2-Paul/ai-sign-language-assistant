import numpy as np
import pytest

from src.vision.hand_detector import HandDetector


def test_detector_returns_no_hands_for_blank_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
        hands = detector.detect(image)
    assert hands == []


def test_detector_rejects_invalid_image():
    with HandDetector(static_image_mode=True) as detector:
        with pytest.raises(ValueError):
            detector.detect(np.array([]))
