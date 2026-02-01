"""
Balanced Model - Natural Learning WITHOUT Aggressive Weights
Strategie: Lăsăm modelul să învețe natural, doar echilibrăm ușor clasele
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

TRAIN_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\train"
VAL_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\validation"
MODEL_SAVE_PATH = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\src\neural_network\saved_models\trained_model_improved.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50

# Data augmentation - moderate, not aggressive
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print("\n📊 Class indices:", train_gen.class_indices)
print(f"📦 Train samples: {train_gen.samples}")
print(f"📦 Val samples: {val_gen.samples}")

# Calculate NATURAL class weights based on inverse frequency
class_counts = {}
for class_name, class_idx in train_gen.class_indices.items():
    class_counts[class_idx] = len(train_gen.classes[train_gen.classes == class_idx])

total_samples = sum(class_counts.values())
class_weights = {}
for class_idx, count in class_counts.items():
    # Gentle balancing: sqrt of inverse frequency
    class_weights[class_idx] = np.sqrt(total_samples / (len(class_counts) * count))

print("\n⚖️ Natural class weights:", {train_gen.class_indices[k]: f"{v:.2f}" for k, v in 
      sorted([(k, class_weights[v]) for k, v in train_gen.class_indices.items()], key=lambda x: x[1])})

# Transfer learning with MobileNetV2
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze most layers, train only top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze all except last 30
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🏗️ Balanced Natural Model:")
print(f"Total params: {model.count_params():,}")
print(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

print("\n🚀 Training with NATURAL balanced weights...")
history = model.fit(
    train_gen, 
    epochs=EPOCHS, 
    validation_data=val_gen,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=12, 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                         min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', 
                       save_best_only=True, verbose=1)
    ],
    class_weight=class_weights
)

best_acc = max(history.history['val_accuracy'])
print(f"\n✅ Model saved! Best val acc: {best_acc:.4f}")
print("✓ Natural class weights (gentle balancing)")
print("✓ MobileNetV2 transfer learning")
print("✓ 30 top layers trainable")
print("✓ Moderate augmentation")
print("\n🎯 Restart Streamlit WITHOUT post-processing boost!")
