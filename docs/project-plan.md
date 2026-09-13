# Project Development Plan

## Phase 1 — Vision foundation
- Set up the Python environment
- Detect hands from images
- Extract 21 normalized landmarks per hand
- Visualize detections
- Add basic automated tests

## Phase 2 — Dataset
- Choose and document the initial sign vocabulary
- Identify suitable public datasets and/or collect controlled samples
- Define train/validation/test splits
- Check class balance and data quality

## Phase 3 — Baseline ML model
- Convert landmarks into model-ready features
- Train a simple baseline classifier
- Evaluate accuracy, precision, recall, F1, and confusion matrix
- Save model artifacts outside Git when they become large

## Phase 4 — Recognition
- Build image-based sign prediction
- Add confidence thresholds
- Handle unknown/low-confidence gestures
- Maintain prediction history

## Phase 5 — Assistant features
- Convert recognized signs into text
- Build basic phrase construction
- Add optional text-to-speech

## Phase 6 — Real-time application
- Add webcam input
- Stabilize predictions across frames
- Improve latency and usability
- Polish the Streamlit interface

## Phase 7 — Portfolio / Fiverr-ready package
- Add architecture documentation
- Add setup and deployment instructions
- Add evaluation results and limitations
- Prepare a clean demo and screenshots
- Provide an explanation suitable for project presentation/viva support

## Current milestone

Phase 1 is in progress. The next technical task is to validate the landmark pipeline with representative hand images before moving into dataset design.
