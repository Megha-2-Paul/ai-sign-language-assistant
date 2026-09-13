# Dataset directory

This directory documents the expected local dataset layout. Large raw and processed datasets should not be committed to Git.

Recommended local structure:

```text
data/
├── raw/
│   ├── train/
│   ├── validation/
│   └── test/
├── processed/
│   └── landmarks.csv
└── README.md
```

## Dataset principles

- Record or use samples from multiple signers where possible.
- Keep labels explicit and consistent.
- Keep train/validation/test separation at the signer level when signer identity is available.
- Record dataset source, license, vocabulary, and preprocessing assumptions.
- Do not commit large datasets or personally identifying information.
