# Dataset setup

Datasets and trained models are intentionally not committed. Download an image dataset whose license permits your intended use and review its attribution and terms first.

Expected layout: one directory per isolated sign label, e.g. `data/raw/asl_alphabet_train/A/*.jpg`, `.../B/*.jpg`, and so on. The Kaggle ASL Alphabet dataset is a common learning benchmark, but check its current terms and ensure folder labels match your target vocabulary.

Run from the repository root:
```bash
python scripts/collect_landmarks.py data/raw/asl_alphabet_train data/processed/landmarks.csv
python scripts/train_model.py
```
Every detected hand yields one row. Images without detectable hands are skipped. Inspect class counts and data quality before training. Keep personal photos, identifying information, and restricted datasets out of Git; data folders are git-ignored.
