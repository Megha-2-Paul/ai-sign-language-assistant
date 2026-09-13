"""Streamlit demo for hand landmark detection."""

from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.hand_detector import HandDetector  # noqa: E402


st.set_page_config(page_title="Hand Landmark Demo", page_icon="✋", layout="wide")
st.title("AI Sign Language Assistant")
st.caption("Milestone 1 — hand detection and landmark extraction")

uploaded_file = st.file_uploader("Upload a hand image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image containing a visible hand to begin.")
    st.stop()

image = np.array(Image.open(uploaded_file).convert("RGB"))
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
    hands = detector.detect(image_bgr)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Detection")
    annotated = image.copy()
    height, width = annotated.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    for hand in hands:
        points = hand.landmarks
        xy = [(int(x * width), int(y * height)) for x, y, _ in points]
        for start, end in connections:
            cv2.line(annotated, xy[start], xy[end], (0, 255, 0), 2)
        for x, y in xy:
            cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
    st.image(annotated, use_container_width=True)

st.subheader("Results")
if not hands:
    st.warning("No hand was detected. Try a clearer image with the hand fully visible.")
else:
    st.success(f"Detected {len(hands)} hand(s).")
    for index, hand in enumerate(hands, start=1):
        confidence = f"{hand.confidence:.3f}" if hand.confidence is not None else "N/A"
        st.write(f"**Hand {index}:** {hand.handedness or 'Unknown'} · confidence {confidence} · 21 landmarks")
