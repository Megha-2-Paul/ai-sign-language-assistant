# Dataset Strategy

## Initial scope

The first model will recognize a deliberately limited set of isolated/static signs or gestures. The final vocabulary will be selected after comparing available datasets and licensing, sample quality, signer diversity, and practical recognition difficulty.

The project will not claim to translate unrestricted sign language. Full sign languages also depend on movement, facial expression, body posture, context, and language-specific grammar.

## Sample requirements

We want variation in:

- Signers and hand sizes
- Camera distance and angle
- Lighting conditions
- Backgrounds
- Hand orientation and position
- Image quality

## Split strategy

Where signer identity is available, train/validation/test splits should be separated by signer. This reduces the risk that nearly identical samples from the same person appear in both training and evaluation data.

A candidate starting split is approximately 70/15/15, adjusted when dataset size or class balance requires it.

## Feature representation

The baseline pipeline uses 21 MediaPipe hand landmarks with x, y, and z coordinates. Coordinates are centered on the wrist and normalized by hand scale before being flattened into an ML feature vector.

## Quality controls

Before training, inspect:

- Class balance
- Missing/corrupt images
- Duplicate or near-duplicate samples
- Landmark detection failures
- Signer distribution
- Lighting/background bias
- Label consistency

## Dataset provenance

For every external dataset, record its name, source, license, vocabulary, citation/attribution requirements, and any preprocessing already applied. Do not commit restricted or personally identifying data.

## Future expansion

After a strong static-sign baseline, investigate temporal features and sequence models for dynamic signs. Only then consider broader phrase-level or continuous recognition.
