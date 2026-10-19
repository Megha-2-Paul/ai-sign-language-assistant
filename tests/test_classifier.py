import numpy as np
import pandas as pd
import pytest

from src.models.classifier import train_classifier


def sample_data():
    rng = np.random.default_rng(7)
    rows, labels, splits = [], [], []
    for label, center in [("A", -1.0), ("B", 1.0), ("C", 3.0)]:
        for index in range(10):
            rows.append(rng.normal(center, .08, size=63))
            labels.append(label)
            splits.append("test" if index >= 8 else "train")
    return (
        pd.DataFrame(rows, columns=[f"f{i}" for i in range(63)]),
        pd.Series(labels),
        pd.Series(splits),
    )


def test_training_returns_metrics_and_model():
    x, y, _ = sample_data()
    bundle, metrics = train_classifier(x, y, n_estimators=20)
    assert bundle["metadata"]["feature_count"] == 63
    assert set(bundle["classes"]) == {"A", "B", "C"}
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["train_samples"] + metrics["test_samples"] == len(y)
    assert metrics["evaluation_method"] == "stratified_random_split"
    assert len(metrics["confusion_matrix"]) == 3


def test_predefined_split_is_used_without_resplitting():
    x, y, split = sample_data()
    bundle, metrics = train_classifier(x, y, split=split, n_estimators=20, test_size=0.5)
    assert metrics["evaluation_method"] == "predefined_train_test_split"
    assert metrics["train_samples"] == 24
    assert metrics["test_samples"] == 6
    assert metrics["split_counts"] == {"train": 24, "test": 6}
    assert bundle["metadata"]["evaluation_method"] == "predefined_train_test_split"


def test_training_rejects_test_label_absent_from_training():
    x, y, split = sample_data()
    y.iloc[0] = "unseen"
    split.iloc[0] = "test"
    with pytest.raises(ValueError, match="absent from training"):
        train_classifier(x, y, split=split, n_estimators=5)


def test_training_rejects_invalid_split_values():
    x, y, split = sample_data()
    split.iloc[0] = "validation"
    with pytest.raises(ValueError, match="split must contain"):
        train_classifier(x, y, split=split, n_estimators=5)


def test_training_rejects_one_class():
    x, y, _ = sample_data()
    with pytest.raises(ValueError, match="two different"):
        train_classifier(x, y.map(lambda _: "A"))


def test_training_rejects_rare_class():
    x, y, _ = sample_data()
    y.iloc[-1] = "rare"
    with pytest.raises(ValueError, match="at least two samples"):
        train_classifier(x, y, n_estimators=5)


def test_training_rejects_invalid_test_size():
    x, y, _ = sample_data()
    with pytest.raises(ValueError, match="test_size"):
        train_classifier(x, y, test_size=1.0)
