"""
CENTRAL CROP MODEL - Train only on CENTRAL regions (remove caps)
Strategy:
1. Custom preprocessing: crop top 20% and bottom 10% (remove caps)
2. Focus on bottle BODY texture (transparent, labels, liquid)
3. MobileNetV2 transfer learning
4. Moderate augmentation (no extreme zoom since we already crop)
"""

import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==================== CUSTOM PREPROCESSING ====================
import cv2

def central_crop_preprocessing(img):
    """
    Crop top 20% and bottom 10% to remove bottle caps and base.
    Focus on central body region where material is most visible.
    Works with numpy arrays (OpenCV style).
    """
    if isinstance(img, np.ndarray):
        height, width = img.shape[:2]
        
        # Calculate crop region
        top_crop = int(height * 0.20)  # Remove top 20%
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
# Moderate augmentation since we already crop
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=central_crop_preprocessing,  # Apply crop before other ops
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=central_crop_preprocessing  # Apply same crop to validation
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

# ==================== CLASS WEIGHTS (sqrt balanced) ====================
class_counts = {}
for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        class_counts[class_name] = count

total_samples = sum(class_counts.values())
n_classes = len(class_counts)

class_weights = {}
print("\n⚖️ Class Weights (sqrt balanced):")
for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
    weight = np.sqrt(total_samples / (n_classes * count))
    class_weights[idx] = weight
    print(f"  {class_name}: {count} images (weight: {weight:.2f})")

# ==================== BUILD MODEL ====================
print("\n🏗️ Building CENTRAL CROP Model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze last 40 layers (moderate fine-tuning)
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
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
print("\n🚀 Training CENTRAL CROP Model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: 0.0003")
print(f"   Trainable layers: {trainable_count}")
print(f"   Crop: top 20% + bottom 10% REMOVED")
print(f"   Focus: CENTRAL BODY (70% middle region)")
print(f"   Class weights: sqrt balanced\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==================== SAVE ====================
print(f"\n✅ CENTRAL CROP Model saved to: {MODEL_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Save history
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"📈 History saved: {HISTORY_PATH}")

print("\n🎯 CENTRAL CROP Model trained!")
print("   ✓ Top 20% removed (caps eliminated)")
print("   ✓ Bottom 10% removed (base eliminated)")
print("   ✓ Focus on central 70% (bottle body)")
print("   ✓ Should classify Coca-Cola as PLASTIC/STICLA!")
print("\n   Restart Streamlit and test now!")
