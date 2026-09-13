# Evaluation Strategy

Model quality will be evaluated beyond a single accuracy number.

## Primary metrics

- Accuracy
- Precision
- Recall
- F1 score
- Per-class performance
- Confusion matrix

## Generalization

When signer metadata is available, keep signers separated between training and evaluation sets. This provides a more realistic estimate of performance on a person the model has not seen.

## Robustness checks

Evaluate representative variation in:

- Lighting
- Background
- Hand position
- Camera distance
- Hand orientation
- Image quality

## Runtime metrics

For application readiness, also record:

- Inference latency
- Frames processed per second for live inference
- Model size
- Memory/CPU requirements where practical

## Reporting

Every trained model should record its dataset version, preprocessing version, model configuration, metrics, and known failure cases. Reported accuracy must come from an explicit held-out evaluation set rather than training performance.
