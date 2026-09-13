# Dataset Strategy

## Objective

The first model will recognize a limited vocabulary of isolated/static hand signs. The project will not claim unrestricted sign-language translation.

## MVP dataset

The initial benchmark is the ASL Alphabet dataset because it provides a practical starting point for building and validating the complete pipeline.

Target classes:

- A-Z
- SPACE
- DELETE
- NOTHING

The dataset is large enough to establish a meaningful baseline while keeping the first model tractable.

## Advanced dataset

ASL Citizen is a candidate for a later isolated-sign recognition version. It is substantially larger and more challenging, with many sign classes and multiple signers. Its licensing and commercial-use terms must be reviewed before any use beyond research/portfolio experimentation.

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
