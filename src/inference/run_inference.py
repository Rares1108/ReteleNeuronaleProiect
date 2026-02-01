# src/inference/run_inference.py
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

CONFIG_PATH = "config/training_config.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

IMG_H = cfg["img_height"]
IMG_W = cfg["img_width"]
MODEL_PATH = cfg["model_path"]

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_W, IMG_H))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def main(image_path):
    if not os.path.exists(MODEL_PATH):
        print("Model not found at", MODEL_PATH)
        sys.exit(1)

    model = load_model(MODEL_PATH)
    # Attempt to recover class indices from training generator saved as json (optional)
    # For now, user must know class order; better: include classes.json in repo.
    # Try to load classes mapping if exists
    classes_file = os.path.join(os.path.dirname(MODEL_PATH), "classes.json")
    classes = None
    if os.path.exists(classes_file):
        import json
        with open(classes_file,"r") as f:
            classes = json.load(f)
            inv_map = {int(v):k for k,v in classes.items()}
    else:
        inv_map = None

    x = load_and_preprocess(image_path)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    if inv_map:
        label = inv_map.get(idx, str(idx))
    else:
        label = str(idx)
    print("Prediction:", label)
    print("Probabilities:", probs.tolist())
    return label, probs.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()
    main(args.image)
