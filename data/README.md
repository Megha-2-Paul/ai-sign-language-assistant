# Dataset setup

Datasets and trained models are intentionally not committed. Before using an external dataset, verify its current licence, terms of use, attribution requirements, and whether redistribution or commercial use is permitted. Keep original dataset files outside Git.

## Recommended dataset: ASL-HG

ASL-HG contains 36,000 static images across 36 classes (A–Z and digits 0–9), collected from 10 volunteers. It is available from [Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1) under CC BY 4.0. Read the dataset description and licence on the source page before downloading. Cite the dataset authors in reports and public demonstrations:

Pranto, M. F. I., Islam, M. R., Akbor, M. A., Ghosh, N., Rahatun, M., Chaki, S., & Masudul, M. (2026). *A comprehensive image dataset of American Sign Language hand gestures*. Data in Brief, 65, 112492. https://doi.org/10.1016/j.dib.2026.112492

**Important modelling limitation:** this application predicts each detected hand independently. The ASL-HG class `0` represents a two-handed sign, so a per-hand classifier cannot correctly model it as one class. For this MVP, exclude `0` rather than assigning the same two-handed label to each individual hand. Supporting two-handed signs requires a paired-hand feature representation and separate evaluation.

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

## Prepare and train

From the repository root:

```bash
# Initial pipeline run: cap images to limit processing time and exclude the two-handed zero sign.
python scripts/collect_landmarks.py data/raw/sign_dataset data/processed/landmarks.csv --max-images-per-class 100 --random-state 42 --exclude-labels 0

# Train/evaluate using predefined train/test folders when the source provides them.
python scripts/train_model.py --random-state 42

# Launch after a model artifact has been produced.
streamlit run app/streamlit_app.py
```

The image cap is applied independently per label and predefined split, using deterministic sampling. Remove `--max-images-per-class` to process all images. The output CSV records `split=train/test` when those folders exist; split metadata is excluded from the 63 model features. If the dataset has no predefined split, the trainer falls back to a stratified random split and records that evaluation method in `models/evaluation.json`.

Every detected hand yields one row; images without detected hands are skipped. Inspect class counts, skipped images, label correctness, and duplicate/near-duplicate risk before interpreting metrics. Keep personal photos, identifying information, and restricted datasets out of Git.
