# Dataset Directory

Use this directory for local datasets and generated features.

```text
data/
├── raw/         # Original downloaded/collected samples
├── processed/   # Cleaned/resized samples when required
└── landmarks/   # Generated landmark feature files
```

Large datasets and generated artifacts should remain local and are excluded from Git where appropriate.

## Dataset record

For every dataset, document:

- Dataset name and source
- License and permitted use
- Citation/attribution requirements
- Classes/vocabulary
- Number of samples
- Signer metadata, if available
- Preprocessing performed
- Train/validation/test split

Do not store private, restricted, or personally identifying data in this repository.
