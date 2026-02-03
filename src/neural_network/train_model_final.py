"""
FINAL MODEL - EXTREME SOLUTION
Strategy:
1. ZERO class weights (treat all equal)
2. Aggressive augmentation for plastic/glass
3. Deeper fine-tuning (50 layers)
4. Higher dropout to ignore small details (caps)
5. Ensemble approach - train on body not caps
"""

import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==================== CONFIG ====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 4

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'
HISTORY_PATH = 'src/neural_network/saved_models/trained_model_improved_history.json'

# ==================== DATA AUGMENTATION ====================
# EXTREME augmentation to focus on BOTTLE BODY not caps
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,        # More rotation
    width_shift_range=0.2,     # More shifts
    height_shift_range=0.2,
    shear_range=0.2,           # Add shear
    zoom_range=0.3,            # More zoom (might crop cap)
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Vary lighting
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

# ==================== NO CLASS WEIGHTS (treat all equal) ====================
print("\n⚖️ NO class weights - treating all classes equally!")
print("   Strategy: Let model learn naturally from augmentation\n")

# ==================== BUILD MODEL ====================
print("🏗️ Building FINAL Model (extreme approach)...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze only first 100 layers (unfreeze MORE)
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"   Trainable layers: {trainable_count}/{len(base_model.layers)}")

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.6),  # HIGHER dropout to ignore small details
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0002),  # Lower LR for stability
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
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ==================== TRAINING ====================
print("\n🚀 Training FINAL Model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: 0.0002")
print(f"   Trainable layers: {trainable_count}")
print(f"   NO class weights (equal treatment)")
print(f"   EXTREME augmentation (zoom/shear to hide caps)")
print(f"   HIGHER dropout (0.6/0.5/0.4)\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ==================== SAVE ====================
print(f"\n✅ FINAL Model saved to: {MODEL_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Save history
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"📈 History saved: {HISTORY_PATH}")

print("\n🎯 This FINAL model uses:")
print("   ✓ NO class weights (equal treatment)")
print("   ✓ EXTREME augmentation (zoom/crop caps)")
print("   ✓ HIGHER dropout (ignore small details)")
print("   ✓ 54 trainable layers (deep fine-tuning)")
print("\n   Test with Coca-Cola now! Should focus on BOTTLE not CAP!")
