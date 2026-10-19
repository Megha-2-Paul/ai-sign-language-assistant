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


def train_classifier(
    features,
    labels,
    *,
    split=None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a Random Forest and evaluate on a predefined or stratified hold-out."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if len(features) != len(labels) or len(labels) < 4:
        raise ValueError("Features and labels must have matching lengths and at least four samples")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain only finite numeric values")

    labels = labels.astype(str).reset_index(drop=True)
    features = features.reset_index(drop=True)
    counts = labels.value_counts()
    if len(counts) < 2:
        raise ValueError("Training requires at least two different sign labels")

    if split is None:
        if (counts < 2).any():
            raise ValueError(
                "Each label needs at least two samples; insufficient samples for: "
                + ", ".join(sorted(counts[counts < 2].index))
            )
        n_classes, n_samples = len(counts), len(labels)
        n_test = int(np.ceil(n_samples * test_size))
        if n_test < n_classes or n_samples - n_test < n_classes:
            raise ValueError("Dataset is too small for a stratified train/test split; add samples per class")
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        evaluation_method = "stratified_random_split"
        split_counts = {"train": int(len(y_train)), "test": int(len(y_test))}
    else:
        split = split.astype(str).str.strip().str.lower().reset_index(drop=True)
        if len(split) != len(labels) or split.isna().any() or not set(split).issubset({"train", "test"}):
            raise ValueError("split must contain one 'train' or 'test' value for every sample")
        if set(split) != {"train", "test"}:
            raise ValueError("Predefined split must contain both train and test samples")
        train_mask, test_mask = split.eq("train"), split.eq("test")
        x_train, x_test = features.loc[train_mask], features.loc[test_mask]
        y_train, y_test = labels.loc[train_mask], labels.loc[test_mask]
        if len(y_train) < 2 or len(y_test) < 1:
            raise ValueError("Predefined split has too few training or test samples")
        train_counts = y_train.value_counts()
        if len(train_counts) < 2:
            raise ValueError("Training split must contain at least two different sign labels")
        if (train_counts < 2).any():
            raise ValueError("Each training label needs at least two samples")
        unknown_test_labels = sorted(set(y_test) - set(y_train))
        if unknown_test_labels:
            raise ValueError("Test split contains labels absent from training: " + ", ".join(unknown_test_labels))
        evaluation_method = "predefined_train_test_split"
        split_counts = {"train": int(len(y_train)), "test": int(len(y_test))}
        counts = labels.value_counts()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    classes = sorted(str(value) for value in unique_labels(y_train))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_count": int(len(classes)),
        "class_distribution": {str(key): int(value) for key, value in sorted(counts.items())},
        "split_counts": split_counts,
        "evaluation_method": evaluation_method,
        "classification_report": classification_report(
            y_test, predictions, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "class_labels": classes,
        "random_state": int(random_state),
        "test_size": float(len(y_test) / (len(y_train) + len(y_test))),
    }
    bundle = {
        "model": model,
        "feature_columns": list(features.columns),
        "classes": classes,
        "metadata": {
            "algorithm": "RandomForestClassifier",
            "feature_count": int(features.shape[1]),
            "landmarks": 21,
            "coordinates_per_landmark": 3,
            "task": "isolated_static_sign_recognition",
            "evaluation_method": evaluation_method,
        },
    }
    return bundle, metrics


def save_model(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a model bundle, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination
