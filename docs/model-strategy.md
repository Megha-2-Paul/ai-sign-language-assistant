# Model Strategy

## Goal

Build a strong, explainable baseline before introducing more computationally expensive models.

## Model progression

### Baseline 1 — Classical ML

Train a classifier on normalized MediaPipe hand landmarks.

Candidates:

- Random Forest
- Support Vector Machine (SVM)

Advantages:

- Fast training
- Low inference cost
- Easy to inspect and explain
- Well suited to landmark features

### Baseline 2 — Small neural network

Compare a compact multilayer perceptron against the classical baseline using the same landmark features.

### Baseline 3 — Image-based CNN

If landmark-only features are insufficient, evaluate a CNN directly on images. This can capture visual information that landmarks discard, but requires more compute and careful regularization.

### Future — Temporal model

For dynamic signs, investigate sequences of landmarks/frames with temporal models. This is a later milestone, not part of the first static-sign baseline.

## Model selection principle

Choose the simplest model that provides strong validation performance, generalizes to unseen signers where possible, and meets real-time inference requirements.
