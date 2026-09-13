# AI Sign Language Recognition & Speech Assistant

A portfolio-grade computer vision project that recognizes a defined vocabulary of sign-language gestures and converts recognized signs into text, with optional text-to-speech output.

## Project goal

Build a working, explainable AI application suitable for demonstrating computer vision, machine learning, and real-time/image-based inference in a university-style project and professional portfolio.

> **Scope note:** The initial version will recognize a limited, explicitly documented vocabulary. It is not intended to claim unrestricted translation of all sign languages.

## Planned capabilities

- Hand detection and landmark extraction
- Sign/gesture classification
- Image-upload recognition as the primary MVP input
- Optional live-camera recognition
- Confidence score for predictions
- Recognized-sign history and phrase construction
- Text-to-speech output
- Streamlit-based user interface
- Model evaluation and documented experiments

## Planned technology stack

- Python
- OpenCV
- MediaPipe
- scikit-learn / deep learning as appropriate
- Streamlit
- Text-to-speech library

## Project structure

```text
ai-sign-language-assistant/
├── app/              # Streamlit application
├── data/              # Dataset documentation and local data instructions
├── models/            # Trained model artifacts (large files excluded from Git)
├── notebooks/         # Experiments and model development
├── src/               # Reusable application and ML code
├── tests/             # Automated tests
├── README.md
└── requirements.txt
```

## Development roadmap

1. Environment and dependency setup
2. Hand landmark detection prototype
3. Dataset collection/preparation
4. Feature extraction and preprocessing
5. Baseline model training
6. Model evaluation and improvement
7. Recognition application
8. Text-to-speech and phrase construction
9. UI polish and testing
10. Portfolio documentation and demo

## Responsible use

Recognition accuracy depends on the selected vocabulary, dataset, lighting, camera/image quality, signer variation, and other factors. Results should be treated as model predictions rather than authoritative interpretation.
