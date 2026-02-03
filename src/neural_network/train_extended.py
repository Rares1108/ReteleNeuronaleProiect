"""
Extended Training - More epochs, better convergence
Strategie: Antrenare mai lungă pentru învățare mai bună a diferențelor între clase
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
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
EPOCHS = 100  # More epochs for better learning

# Strong augmentation to learn differences
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
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

# Balanced class weights with EMPHASIS on metal (not extreme)
class_counts = {}
for class_name, class_idx in train_gen.class_indices.items():
    class_counts[class_idx] = len(train_gen.classes[train_gen.classes == class_idx])

total_samples = sum(class_counts.values())
class_weights = {}
for class_idx, count in class_counts.items():
    class_weights[class_idx] = np.sqrt(total_samples / (len(class_counts) * count))

# Boost metal slightly more (1.5x of natural weight)
class_weights[1] *= 1.5  # metal class

print("\n⚖️ Class weights:", {k: f"{class_weights[train_gen.class_indices[k]]:.2f}" 
      for k in sorted(train_gen.class_indices.keys())})

# MobileNetV2 with more trainable layers
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Train top 40 layers (more than before)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.4),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.35),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n🏗️ Extended Training Model:")
print(f"Total params: {model.count_params():,}")
print(f"Trainable params: {trainable:,}")
print(f"Trainable layers: {sum([1 for l in model.layers if l.trainable])}")

print("\n🚀 Training with EXTENDED epochs (up to 100)...")
history = model.fit(
    train_gen, 
    epochs=EPOCHS, 
    validation_data=val_gen,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=20, 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, 
                         min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', 
                       save_best_only=True, verbose=1)
    ],
    class_weight=class_weights
)

best_acc = max(history.history['val_accuracy'])
print(f"\n✅ Model saved! Best val acc: {best_acc:.4f}")
print(f"✓ Metal weight: {class_weights[1]:.2f}x (balanced + 1.5x boost)")
print("✓ 40 layers trainable")
print("✓ Strong augmentation")
print("✓ Extended training (patience=20)")
