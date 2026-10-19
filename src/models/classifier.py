"""Train, evaluate, and persist a reproducible baseline sign classifier."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

def train_classifier(features, labels, *, test_size: float = 0.2, random_state: int = 42, n_estimators: int = 300) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a random forest and return a serializable bundle plus held-out metrics."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if len(features) != len(labels) or len(labels) < 4:
        raise ValueError("Features and labels must have matching lengths and at least four samples")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain only finite numeric values")
    labels = labels.astype(str)
    counts = labels.value_counts()
    if len(counts) < 2:
        raise ValueError("Training requires at least two different sign labels")
    if (counts < 2).any():
        raise ValueError("Each label needs at least two samples; insufficient samples for: " + ", ".join(sorted(counts[counts < 2].index)))
    n_classes, n_samples = len(counts), len(labels)
    n_test = int(np.ceil(n_samples * test_size))
    if n_test < n_classes or n_samples - n_test < n_classes:
        raise ValueError("Dataset is too small for a stratified train/test split; add samples per class")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    classes = sorted(str(v) for v in unique_labels(labels))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "test_samples": int(len(y_test)), "train_samples": int(len(y_train)), "class_count": int(n_classes),
        "class_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
        "classification_report": classification_report(y_test, predictions, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "class_labels": classes, "random_state": int(random_state), "test_size": float(test_size),
    }
    bundle = {
        "model": model, "feature_columns": list(features.columns), "classes": classes,
        "metadata": {"algorithm": "RandomForestClassifier", "feature_count": int(features.shape[1]), "landmarks": 21, "coordinates_per_landmark": 3, "task": "isolated_static_sign_recognition"},
    }
    return bundle, metrics

def save_model(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a model bundle, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination
