# AI Sign Language Assistant

A portfolio-focused Python computer-vision application for **isolated static sign recognition** from uploaded images. It uses MediaPipe hand landmarks, a scikit-learn classifier, and Streamlit with confidence estimates, phrase building, session history, and browser speech.

> **Scope:** This is a limited-vocabulary recognition assistant, not a continuous sign-language translator. It does not understand grammar, facial expression, or movement-based signs. Predictions may be wrong and must not be treated as authoritative interpretation.

## Features

- Image-upload inference; the primary app does not open a webcam.
- MediaPipe extraction of 21 hand landmarks and consistent 63-feature normalization.
- Reproducible Random Forest training and stratified held-out evaluation.
- Dataset validation; capture metadata is excluded from model features.
- Predictions, estimated confidence, phrase builder, undo/clear, session-only history, browser text-to-speech.
- Evaluation JSON with accuracy, class counts, per-class metrics, and confusion matrix.
- Automated tests and GitHub Actions CI.

## Quick start (Python 3.11)

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```
macOS/Linux: create a virtual environment with `python3.11 -m venv .venv`, activate it, and run `python -m pip install -r requirements-dev.txt`.

## Dataset and training

No dataset or trained model is committed. Download a dataset that permits your intended use and review its license and attribution requirements. See [dataset setup](data/README.md).

Expected layout: `data/raw/asl_alphabet_train/<LABEL>/*.jpg`.

```bash
python scripts/collect_landmarks.py data/raw/asl_alphabet_train data/processed/landmarks.csv
python scripts/train_model.py --random-state 42 --test-size 0.2
```

The first command extracts landmarks. The second saves `models/sign_classifier.joblib` and `models/evaluation.json`. Training fails clearly when data is missing, malformed, too small, or lacks enough samples per class.

## Run the app

After training, run `streamlit run app/streamlit_app.py`. Upload a clear image containing one or two visible hands, add predicted labels to the phrase, undo or clear labels, inspect recent session history, or use browser speech synthesis. If the model artifact is missing, the app displays commands to create it.

The diagnostic landmark-only interface remains available with `streamlit run app/landmark_demo.py`.

## Tests

Run `python -m pytest -q`. GitHub Actions uses the same test suite on Python 3.11 for pushes and pull requests.

## Project layout

- `src/vision/`: MediaPipe hand detection.
- `src/preprocessing/`: landmark normalization and features.
- `src/data/`: dataset loading and validation.
- `src/models/`: training, evaluation, serialization.
- `src/inference/`: model loading and prediction.
- `scripts/`: data collection and training CLIs.
- `app/`: recognition and diagnostic Streamlit interfaces.
- `tests/`: regression tests.
- `docs/model-evaluation.md`: evaluation interpretation and limitations.

## Responsible use

Performance depends on dataset quality, label coverage, lighting, image quality, hand visibility, and signer variation. Do not report accuracy until training and evaluation have run on a documented dataset. Prefer signer-independent test data when possible. Dynamic signs and phrase-level translation are outside this baseline's scope.
