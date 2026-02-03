"""
REDUCED METAL BIAS MODEL
Strategy:
1. Central crop (remove caps) - already working
2. REDUCE metal class weight to 0.5 (penalize metal predictions)
3. INCREASE plastic/glass weights to 1.5 (favor these)
4. Add color-based augmentation (metal objects don't have colorful labels)
5. MobileNetV2 with moderate fine-tuning
"""

import os
import numpy as np
import json
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==================== CUSTOM PREPROCESSING ====================
def central_crop_preprocessing(img):
    """
    Crop top 35% and bottom 10% to COMPLETELY remove bottle caps and base.
    """
    if isinstance(img, np.ndarray):
        height, width = img.shape[:2]
        
        # Calculate crop region
        top_crop = int(height * 0.35)  # Remove top 35% (was 20% - not enough!)
        bottom_crop = int(height * 0.10)  # Remove bottom 10%
        
        # Crop central region (70% of image - middle section)
        cropped = img[top_crop:height-bottom_crop, :, :]
        
        # Resize back to target size using OpenCV
        cropped_resized = cv2.resize(cropped, (128, 128))
        
        return cropped_resized
    return img

# ==================== CONFIG ====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
NUM_CLASSES = 4

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'
HISTORY_PATH = 'src/neural_network/saved_models/trained_model_improved_history.json'

# ==================== DATA AUGMENTATION ====================
# Enhanced color augmentation - metal objects typically don't have colorful labels
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=central_crop_preprocessing,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Wider range
    channel_shift_range=30.0,  # Color shifts - helps distinguish metal
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=central_crop_preprocessing
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\n📊 Class Distribution:")
class_indices = train_gen.class_indices
for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
    count = len([f for f in os.listdir(os.path.join(TRAIN_DIR, class_name)) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"  {class_name}: {count} images")

# ==================== CUSTOM CLASS WEIGHTS (REDUCE METAL!) ====================
class_counts = {}
for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        class_counts[class_name] = count

# Manual weights - penalize metal moderately, favor plastic/glass
class_weights = {
    0: 0.85,  # hartie - normal
    1: 0.65,  # metal - MODERATELY REDUCED (not too aggressive)
    2: 1.15,  # plastic - INCREASED
    3: 1.15   # sticla - INCREASED
}

print("\n⚖️ CUSTOM Class Weights (Metal Penalized):")
for idx, class_name in enumerate(['hartie', 'metal', 'plastic', 'sticla']):
    count = class_counts.get(class_name, 0)
    weight = class_weights[idx]
    print(f"  {class_name}: {count} images (weight: {weight:.2f})")

# ==================== BUILD MODEL ====================
print("\n🏗️ Building REDUCED METAL BIAS Model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze last 35 layers (moderate)
for layer in base_model.layers[:-35]:
    layer.trainable = False
for layer in base_model.layers[-35:]:
    layer.trainable = True

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"   Trainable layers: {trainable_count}/{len(base_model.layers)}")

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(384, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(192, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# ==================== CALLBACKS ====================
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# ==================== TRAINING ====================
print("\n🚀 Training REDUCED METAL BIAS Model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: 0.0003")
print(f"   Trainable layers: {trainable_count}")
print(f"   Central crop: TOP 25% + BOTTOM 10% removed (65% central region - OPTIMAL)")
print(f"   Metal weight: 0.65 (MODERATELY REDUCED)")
print(f"   Plastic/Glass weight: 1.15 (INCREASED)")
print(f"   Color augmentation: Enhanced (channel shift)\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==================== SAVE ====================
print(f"\n✅ REDUCED METAL BIAS Model saved to: {MODEL_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Save history
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"📈 History saved: {HISTORY_PATH}")

print("\n🎯 REDUCED METAL BIAS Model trained!")
print("   ✓ Central crop applied (caps removed)")
print("   ✓ Metal weight: 0.4 (strongly penalized)")
print("   ✓ Plastic/Glass: 1.2 (favored)")
print("   ✓ Enhanced color augmentation")
print("\n   Coca-Cola should now classify as PLASTIC/STICLA!")
print("   Restart Streamlit and test!")
