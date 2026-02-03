"""
Simple CNN + Aggressive Metal Boost
Strategie: Nu transfer learning, CNN custom trainabil pe toți parametrii
Metal boost: 3x (300%)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path

TRAIN_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\train"
VAL_DIR = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\data\validation"
MODEL_SAVE_PATH = r"C:\Users\rares\Desktop\ReteleNeuronaleProiect-main\src\neural_network\saved_models\trained_model_improved.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
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

# AGGRESSIVE class weights for METAL
class_weights = {
    0: 0.70,   # hartie - penalized
    1: 2.50,   # metal - HEAVILY BOOSTED (was 1.3, now 2.5)
    2: 1.00,   # plastic
    3: 1.00    # sticla
}

print("⚖️ Class weights:", class_weights)

# Simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🏗️ Simple CNN Model:")
model.summary()

print("\n🚀 Training with METAL BOOST 2.5x...")
history = model.fit(
    train_gen, epochs=EPOCHS, validation_data=val_gen,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=10, 
                     restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', 
                       save_best_only=True, verbose=1)
    ],
    class_weight=class_weights
)

print(f"\n✅ Model saved! Best val acc: {max(history.history['val_accuracy']):.4f}")
print("✓ Metal weight: 2.5x (HEAVILY BOOSTED)")
print("✓ Simple CNN (fully trainable)")
print("✓ Strong augmentation")
print("✓ Restart Streamlit to test!")
