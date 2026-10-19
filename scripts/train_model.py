"""CLI for training and evaluating the static-sign baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import load_landmark_csv
from src.models.classifier import save_model, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign classifier from extracted landmark CSV data.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/sign_classifier.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/evaluation.json"))
    parser.add_argument("--test-size", type=float, default=0.2, help="Used only when the CSV has no predefined split")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    try:
        features, labels, split = load_landmark_csv(args.dataset, return_split=True)
        bundle, metrics = train_classifier(
            features,
            labels,
            split=split,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.trees,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    metrics["dataset"], metrics["model"] = str(args.dataset), str(args.model_out)
    save_model(bundle, args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved: {args.model_out}")
    print(f"Evaluation saved: {args.metrics_out}")
    print(f"Evaluation method: {metrics['evaluation_method']}")
    print(f"Held-out accuracy: {metrics['accuracy']:.3f} ({metrics['test_samples']} samples)")
    print("Accuracy is meaningful only when the dataset and evaluation split represent intended real-world use.")


if __name__ == "__main__":
    main()
