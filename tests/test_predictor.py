import numpy as np
import pytest
from src.inference.predictor import predict_landmarks

class DummyClassifier:
    def predict(self, x):
        assert x.shape == (1, 63)
        return np.array(["A"])
    def predict_proba(self, x):
        return np.array([[.85, .15]])

def test_prediction_returns_label_and_confidence():
    bundle = {"model": DummyClassifier(), "feature_columns": [f"f{i}" for i in range(63)], "classes": ["A", "B"]}
    result = predict_landmarks(np.zeros((21, 3), dtype=np.float32), bundle)
    assert result["label"] == "A"
    assert result["confidence"] == pytest.approx(.85)

def test_prediction_rejects_bad_landmark_shape():
    bundle = {"model": DummyClassifier(), "feature_columns": [f"f{i}" for i in range(63)], "classes": ["A", "B"]}
    with pytest.raises(ValueError, match="shape"):
        predict_landmarks(np.zeros((20, 3)), bundle)
