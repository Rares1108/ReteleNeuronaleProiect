import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "..", "neural_network", "saved_models", "trained_model_improved.h5"
)

CLASS_NAMES = ["hartie", "metal", "plastic", "sticla"]  # Alphabetical order (ImageDataGenerator default)
IMG_SIZE = (128, 128)  # Updated to match new model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("♻️ Waste Classification")
st.write("Clasificare deșeuri folosind CNN")

uploaded_file = st.file_uploader(
    "Încarcă o imagine", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagine încărcată", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    
    # NO post-processing - let model decide naturally
    idx = np.argmax(preds[0])
    confidence = preds[0][idx]

    st.success(f"Predicție: {CLASS_NAMES[idx]}")
    st.info(f"Încredere: {confidence * 100:.2f}%")
    
    # Show all class probabilities
    with st.expander("🔍 Detalii scoruri"):
        cols = st.columns(4)
        for i, class_name in enumerate(CLASS_NAMES):
            with cols[i]:
                st.metric(class_name.upper(), f"{preds[0][i]*100:.1f}%")
