import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2
from PIL import Image

# Page config
st.set_page_config(page_title="Waste Classification", layout="wide")

st.title("🗑️ Waste Classification Model")
st.write("Upload an image to classify waste into: Plastic, Paper, Glass, or Metal")

# Load model
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'  # Modelul îmbunătățit
IMG_TARGET_SIZE = (128, 128)  # Rezoluție crescută
CLASS_NAMES = ['hartie', 'metal', 'plastic', 'sticla']  # ALPHABETICAL ORDER (matches ImageDataGenerator!)

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH)

model = load_trained_model()

def central_crop_preprocessing(img_array):
    """
    Crop top 35% and bottom 10% to COMPLETELY remove bottle caps and base.
    Focus on central body region where material is most visible.
    This matches the training preprocessing!
    """
    height, width = img_array.shape[:2]
    
    # Calculate crop region (same as training)
    top_crop = int(height * 0.35)  # Remove top 35% (INCREASED to remove caps completely!)
    bottom_crop = int(height * 0.10)  # Remove bottom 10%
    
    # Crop central region (55% of image - middle section)
    cropped = img_array[top_crop:height-bottom_crop, :, :]
    
    # Resize back to target size
    cropped = cv2.resize(cropped, IMG_TARGET_SIZE)
    
    return cropped

def remove_white_background(img_array):
    """Remove white/bright background by focusing on darker regions"""
    # Convert to grayscale
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Threshold to find non-white regions (darker pixels)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Apply mask to original image
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    
    # Replace background with neutral gray
    result[mask == 0] = [128, 128, 128]  # Gray background
    
    return result

if model is not None:
    # Add preprocessing options
    st.sidebar.subheader("🎛️ Preprocessing Options")
    use_central_crop = st.sidebar.checkbox("✂️ Central Crop (remove caps)", value=True, 
                                           help="Remove top 20% and bottom 10% to focus on bottle body")
    use_bg_removal = st.sidebar.checkbox("🎯 Remove White Background", value=False,
                                         help="Remove white/bright backgrounds")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        img = image.load_img(uploaded_file, target_size=IMG_TARGET_SIZE)  # 128x128
        img_array = image.img_to_array(img)
        
        # Apply central crop if enabled (RECOMMENDED - matches training!)
        if use_central_crop:
            img_array = central_crop_preprocessing(img_array)
            st.sidebar.success("✅ Central crop applied (cap removed)")
        
        # Apply background removal if enabled
        if use_bg_removal:
            img_array = remove_white_background(img_array)
            st.sidebar.success("✅ Background removal applied")
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        with col2:
            st.subheader("Rezultat Clasificare")
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Show only final result
            st.success(f"Obiectul este: **{predicted_class.upper()}**")
            st.info(f"Încredere: {confidence:.1f}%")
else:
    st.error("Failed to load the model. Please make sure it's trained.")
