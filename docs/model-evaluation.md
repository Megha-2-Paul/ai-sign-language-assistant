# Training and evaluation

The baseline is a scikit-learn Random Forest using 63 MediaPipe features (21 landmarks × x/y/z). Coordinates are wrist-centered and scaled by maximum distance from the wrist. Capture metadata such as handedness and detector confidence is excluded from model features.

Run `python scripts/collect_landmarks.py data/raw/asl_alphabet_train data/processed/landmarks.csv`, then `python scripts/train_model.py --random-state 42 --test-size 0.2`. The CLI saves `models/sign_classifier.joblib` and `models/evaluation.json`. Metrics include held-out accuracy, class distribution, per-class precision/recall/F1, and a confusion matrix.

The split is stratified and reproducible, but random image-level splitting is not signer-independent. Near-duplicate images or multiple samples from the same signer can inflate results. Where signer metadata exists, use signer-independent evaluation before claiming generalization. Confidence is an estimated class probability, not a calibrated guarantee.

Limits: isolated static signs only; no continuous signing, grammar, facial expression, or motion-sequence understanding. Results depend on hand detection, lighting, occlusion, background, image quality, signer variation, and dataset coverage. No accuracy claim is made until training and evaluation have actually run on a documented dataset.
