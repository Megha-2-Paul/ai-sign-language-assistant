"""Image-upload sign recognition app. This app never opens a webcam."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference.predictor import load_model_bundle, predict_image  # noqa: E402
MODEL_PATH = ROOT / "models" / "sign_classifier.joblib"
st.set_page_config(page_title="AI Sign Language Assistant", page_icon="🤟", layout="wide")
st.title("🤟 AI Sign Language Assistant")
st.caption("Static sign recognition from an uploaded image · Portfolio MVP")
st.info("This prototype classifies isolated signs from a limited vocabulary. It is not a continuous sign-language translator, and predictions may be wrong. No webcam is accessed.")
if not MODEL_PATH.is_file():
    st.warning("A trained model is not installed yet. Follow the README dataset setup and training steps, then restart this app.")
    st.code("python scripts/collect_landmarks.py data/raw/asl_alphabet_train data/processed/landmarks.csv\npython scripts/train_model.py")
    st.stop()

@st.cache_resource
def get_bundle(path: str, modified_time: float):
    return load_model_bundle(path)

try:
    bundle = get_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()
uploaded = st.file_uploader("Upload an image containing one or two visible hands", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    try:
        image = np.array(Image.open(uploaded).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        left, right = st.columns(2)
        with left:
            st.subheader("Input image")
            st.image(image, use_container_width=True)
        predictions = predict_image(image_bgr, bundle)
        with right:
            st.subheader("Recognition")
            if not predictions:
                st.warning("No hand was detected. Try a clearer image with the hand fully visible.")
            else:
                st.caption("Confidence is the model's estimated class probability, not a guarantee of correctness.")
                for item in predictions:
                    confidence = item["confidence"]
                    confidence_text = f"{confidence:.1%}" if confidence is not None else "Not available"
                    st.metric(f"Hand {item['hand_index']} · {item['handedness'] or 'side unknown'}", item["label"], f"Model confidence: {confidence_text}")
                recognized = " ".join(item["label"] for item in predictions)
                if st.button("Add recognized sign(s) to phrase", type="primary"):
                    st.session_state.setdefault("phrase", [])
                    st.session_state["phrase"].extend(item["label"] for item in predictions)
                    st.session_state.setdefault("history", [])
                    st.session_state["history"].insert(0, {"phrase": recognized})
                    st.session_state["history"] = st.session_state["history"][:20]
                    st.rerun()
    except Exception as exc:
        st.error(f"Image processing failed: {exc}")
phrase = st.session_state.get("phrase", [])
st.divider()
st.subheader("Phrase builder")
st.write(" ".join(phrase) if phrase else "Your recognized signs will appear here.")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("Undo last sign", disabled=not phrase):
        st.session_state["phrase"] = phrase[:-1]
        st.rerun()
with c2:
    if st.button("Clear phrase", disabled=not phrase):
        st.session_state["phrase"] = []
        st.rerun()
with c3:
    if phrase:
        safe_text = json.dumps(" ".join(phrase)).replace("</", "<\\/")
        components.html("<button style='padding:0.45rem 0.8rem;cursor:pointer' onclick='speakPhrase()'>🔊 Speak phrase</button><script>function speakPhrase(){const u=new SpeechSynthesisUtterance(" + safe_text + ");window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}</script>", height=48)
with st.expander("Recent recognition history"):
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No predictions recorded in this session yet.")
    else:
        for entry in history:
            st.write(f"• {entry['phrase']}")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
st.caption("Session history stays in memory for this browser session and is not sent to a server by this app.")
