import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ---------------- CONFIG ----------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
NUM_CLASSES = 4

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'

# ------------- DATA GENERATORS (BALANCED) ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.25,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
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

# Class weights - MAI BALANCED
class_counts = {}
for class_name in train_gen.class_indices:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    class_counts[train_gen.class_indices[class_name]] = len(os.listdir(class_dir))

total = sum(class_counts.values())
# Reduce extreme weights pentru balans mai bun
class_weights = {cls: np.sqrt(total / (len(class_counts) * count)) for cls, count in class_counts.items()}

print("\n📊 BALANCED Class Distribution:")
for cls, count in class_counts.items():
    class_name = list(train_gen.class_indices.keys())[cls]
    print(f"  {class_name}: {count} images (weight: {class_weights[cls]:.2f})")

# ---------------- MODEL BALANCED -----------------
print("\n🏗️ Building BALANCED Model (sweet spot)...")

# MobileNetV2 base
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze doar ultimele 30 layere (sweet spot între frozen și full fine-tune)
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"   Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}/{ len(base_model.layers)}")

# Build model cu L2 regularization pentru stabilitate
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    
    # Dense layers cu L2 regularization
    layers.Dense(384, activation='relu', kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(192, activation='relu', kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile cu learning rate MEDIU (sweet spot)
model.compile(
    optimizer=Adam(learning_rate=0.0003),  # Sweet spot între 0.001 și 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# ---------------- CALLBACKS -----------------
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# --------------- TRAIN ------------------
print("\n🚀 Training BALANCED Model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: 0.0003 (sweet spot)")
print(f"   Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
print(f"   L2 Regularization: 0.01")
print(f"   Class weights: BALANCED (sqrt)\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# --------------- SAVE -------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"\n✅ BALANCED Model saved to: {MODEL_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Save history
import json
history_path = MODEL_PATH.replace('.h5', '_history.json')
with open(history_path, 'w') as f:
    json.dump({
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }, f, indent=2)

print(f"📈 History saved: {history_path}")
print("\n🎯 This model should have BALANCED performance across ALL classes!")
