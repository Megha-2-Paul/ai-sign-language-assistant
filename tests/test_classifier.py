import numpy as np
import pandas as pd
import pytest
from src.models.classifier import train_classifier

def sample_data():
    rng = np.random.default_rng(7)
    rows, labels = [], []
    for label, center in [("A", -1.0), ("B", 1.0), ("C", 3.0)]:
        for _ in range(10):
            rows.append(rng.normal(center, .08, size=63))
            labels.append(label)
    return pd.DataFrame(rows, columns=[f"f{i}" for i in range(63)]), pd.Series(labels)

def test_training_returns_metrics_and_model():
    x, y = sample_data()
    bundle, metrics = train_classifier(x, y, n_estimators=20)
    assert bundle["metadata"]["feature_count"] == 63
    assert set(bundle["classes"]) == {"A", "B", "C"}
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["train_samples"] + metrics["test_samples"] == len(y)
    assert len(metrics["confusion_matrix"]) == 3

def test_training_rejects_one_class():
    x, y = sample_data()
    with pytest.raises(ValueError, match="two different"):
        train_classifier(x, y, y.map(lambda _: "A"))

def test_training_rejects_rare_class():
    x, y = sample_data()
    y.iloc[-1] = "rare"
    with pytest.raises(ValueError, match="at least two samples"):
        train_classifier(x, y, n_estimators=5)

def test_training_rejects_invalid_test_size():
    x, y = sample_data()
    with pytest.raises(ValueError, match="test_size"):
        train_classifier(x, y, test_size=1.0)
