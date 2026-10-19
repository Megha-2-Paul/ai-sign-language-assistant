# Dataset setup

Datasets and trained models are intentionally not committed. Before using any external dataset, verify its current licence, terms of use, attribution requirements, and whether redistribution or commercial use is permitted. Keep the original dataset files outside Git.

## Supported folder layouts

A dataset with a predefined train/test split should use:

```text
data/raw/sign_dataset/
├── train/
│   ├── A/
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...
```

Each label folder contains images for one isolated static sign. The collector also supports an unsplit layout such as `data/raw/sign_dataset/A/*.jpg`; in that case it leaves split metadata blank and the trainer uses a reproducible stratified random split.

## Dataset source and attribution

ASL-HG is a candidate dataset for this project (letters A–Z and digits 0–9). Review the original dataset page and its CC BY 4.0 terms before downloading or using it, and include the required attribution for any public report or demonstration. Do not assume that a dataset licence automatically covers every third-party file, subject, or use case. If the source does not provide a genuine held-out split, do not invent one by placing duplicates or near-duplicates into separate partitions.

## Prepare and train

From the repository root:

```bash
# Optional: limit processing time during an initial pipeline smoke test.
python scripts/collect_landmarks.py data/raw/sign_dataset data/processed/landmarks.csv --max-images-per-class 100 --random-state 42

# Train/evaluate using predefined train/test folders when the source provides them.
python scripts/train_model.py --random-state 42

# Launch after a model artifact has been produced.
streamlit run app/streamlit_app.py
```

The image cap is applied independently per label and predefined split, using deterministic sampling. Remove `--max-images-per-class` to process all images. The output CSV records `split=train/test` when those folders exist; split metadata is excluded from the 63 model features. If the dataset has no predefined split, the trainer falls back to a stratified random split and records that evaluation method in `models/evaluation.json`.

Every detected hand yields one row; images without detected hands are skipped. Inspect class counts, skipped images, label correctness, and duplicate/near-duplicate risk before interpreting metrics. Keep personal photos, identifying information, and restricted datasets out of Git.
