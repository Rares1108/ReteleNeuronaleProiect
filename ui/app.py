# ui/app.py
import streamlit as st
from PIL import Image
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

# config
CONFIG_PATH = "config/training_config.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

IMG_H = cfg["img_height"]
IMG_W = cfg["img_width"]
MODEL_PATH = cfg["model_path"]
CLASSES_JSON = os.path.join(os.path.dirname(MODEL_PATH), "classes.json")

st.set_page_config(page_title="SmartRecycleNet - Inference", layout="centered")

st.title("SmartRecycleNet — Inferență reală")
st.markdown("Încarcă o imagine cu un deșeu și vezi predicția modelului antrenat.")

# load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Modelul nu există la {MODEL_PATH}. Rulează mai întâi antrenarea.")
else:
    model = load_model(MODEL_PATH)
    # load classes mapping if exists
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r") as f:
            classes = json.load(f)
            inv_map = {int(v):k for k,v in classes.items()}
    else:
        inv_map = None

    uploaded = st.file_uploader("Alege o imagine JPG/PNG", type=["jpg","jpeg","png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Imagine încărcată", use_column_width=True)
        st.write("Pregătesc imaginea pentru model...")
        img = image.resize((IMG_W, IMG_H))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        with st.spinner("Rulez inferența..."):
            probs = model.predict(x)[0]
            idx = int(np.argmax(probs))
            if inv_map:
                label = inv_map.get(idx, str(idx))
            else:
                label = str(idx)
            st.success(f"Predicție: {label}")
            st.write("Probabilități (fiecare clasă):")
            # display table
            class_names = [inv_map[i] for i in sorted(inv_map)] if inv_map else list(range(len(probs)))
            import pandas as pd
            df = pd.DataFrame({"class": class_names, "probability": probs})
            st.table(df)
