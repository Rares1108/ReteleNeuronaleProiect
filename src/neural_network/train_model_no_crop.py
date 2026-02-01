"""
Train model WITHOUT central crop - ONLY class weight balancing
Previous attempts:
- 20% crop: 70.7% acc (good but cap still visible)
- 25% crop: 58.6% acc (lost too much context)
- 35% crop: 59.7% acc (lost too much context)

TEST WITHOUT CAP proved: 98.64% PLASTIC with weight balancing alone!

Solution: Keep FULL image + strong weight balancing
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path

# Paths
TRAIN_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\train"
VAL_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\validation"
MODEL_SAVE_PATH = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\src\neural_network\saved_models\trained_model_improved.h5"
HISTORY_PATH = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\src\neural_network\saved_models\trained_model_improved_history.json"

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0003

# Data augmentation WITHOUT crop
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0,  # Color augmentation
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate class distribution
class_counts = {}
for class_name in train_generator.class_indices:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    class_counts[class_name] = count

print("\n📊 Class Distribution:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} images")

# STRONG class weight balancing (no crop needed!)
# metal: strongly penalized (red cans bias)
# plastic/glass: favored (transparent bottles)
class_weights = {
    0: 0.80,  # hartie - slightly reduced
    1: 0.50,  # metal - STRONGLY PENALIZED (was 0.65)
    2: 1.25,  # plastic - STRONGLY FAVORED (was 1.15)
    3: 1.25   # sticla - STRONGLY FAVORED (was 1.15)
}

print("\n⚖️ STRONG Class Weights (Metal Heavily Penalized):")
for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    weight = class_weights[class_idx]
    print(f"  {class_name}: {class_counts[class_name]} images (weight: {weight})")

# Build model
print("\n🏗️ Building NO-CROP Model (Full Image + Strong Weights)...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers, train later ones
for layer in base_model.layers[:-35]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(384, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(192, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

trainable_count = sum([1 for layer in model.layers if layer.trainable and len(layer.trainable_weights) > 0])
total_count = len(base_model.layers)
print(f"Trainable layers: {trainable_count}/{total_count}")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
print("\n📋 Model Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n🚀 Training NO-CROP Model (Full Image)...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Trainable layers: {trainable_count}")
print(f"   NO CROP: Full image preserved (all context: caps, labels, shape)")
print(f"   Metal weight: 0.50 (STRONGLY PENALIZED)")
print(f"   Plastic/Glass weight: 1.25 (STRONGLY FAVORED)")
print(f"   Color augmentation: Enhanced (channel shift)")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save history
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)

print(f"\n✅ NO-CROP Model saved to: {MODEL_SAVE_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"📈 History saved: {HISTORY_PATH}")

print("\n🎯 NO-CROP Model trained!")
print("   ✓ Full image preserved (no crop)")
print("   ✓ Metal weight: 0.5 (strongly penalized)")
print("   ✓ Plastic/Glass: 1.25 (strongly favored)")
print("   ✓ Enhanced color augmentation")
print("\n   Coca-Cola WITH CAP should classify as PLASTIC/STICLA!")
print("   Test on Streamlit interface!")
