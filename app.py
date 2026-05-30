import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawDetect",
    page_icon="🐾",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 32px;
}
.result-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 24px;
    border: 1px solid #2a2a2a;
    background: #161616;
}
.result-animal {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}
.cat  { color: #f97316; }
.dog  { color: #38bdf8; }
.conf { color: #888; font-size: 0.9rem; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🐾 PawDetect</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a photo — is it a cat or a dog?</div>', unsafe_allow_html=True)

if model is None:
    st.error(
        f"⚠️ Model file `{MODEL_PATH}` not found in the project folder.\n\n"
        "Add this line at the end of your notebook and re-run it:\n\n"
        "```python\nmodel.save('cat_dog_model.h5')\n```"
    )
    st.stop()

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("🔍 Identify Animal", use_container_width=True):
        with st.spinner("Analysing…"):
            # Preprocess
            img_resized = img.resize((128, 128))
            arr = np.array(img_resized) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr, verbose=0)[0][0]

            # class_indices: {'Cat': 0, 'Dog': 1}
            if pred >= 0.5:
                label, conf, emoji, css = "Dog", pred * 100, "🐶", "dog"
            else:
                label, conf, emoji, css = "Cat", (1 - pred) * 100, "🐱", "cat"

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:3.5rem">{emoji}</div>
            <div class="result-animal {css}">{label}</div>
            <div class="conf">{conf:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf))
