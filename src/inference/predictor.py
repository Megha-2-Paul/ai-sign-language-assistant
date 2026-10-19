"""Image-to-landmark-to-label inference helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from src.preprocessing.landmarks import flatten_landmarks
from src.vision.hand_detector import HandDetector

def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a locally trained artifact and validate its structure."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Prepare labeled images and run scripts/train_model.py first.")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"model", "feature_columns", "classes"}.issubset(bundle):
        raise ValueError("Invalid model artifact: expected model, feature_columns, and classes")
    if not hasattr(bundle["model"], "predict"):
        raise ValueError("Invalid model artifact: classifier has no predict method")
    if len(bundle["feature_columns"]) != 63:
        raise ValueError("Invalid model artifact: expected 63 landmark features")
    return bundle

def predict_landmarks(landmarks: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    """Classify one hand and return its label and estimated class probability."""
    vector = flatten_landmarks(landmarks)
    row = pd.DataFrame([vector], columns=bundle["feature_columns"])
    model = bundle["model"]
    label = str(model.predict(row)[0])
    predict_proba = getattr(model, "predict_proba", None)
    confidence = float(np.max(model.predict_proba(row)[0])) if callable(predict_proba) else None
    return {"label": label, "confidence": confidence}

def predict_image(image_bgr: np.ndarray, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect up to two hands in a BGR image and predict each static sign."""
    with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
        hands = detector.detect(image_bgr)
    results = []
    for index, hand in enumerate(hands, start=1):
        prediction = predict_landmarks(hand.landmarks, bundle)
        prediction.update({"hand_index": index, "handedness": hand.handedness, "hand_detection_confidence": hand.confidence})
        results.append(prediction)
    return results
