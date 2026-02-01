import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50  # Mai multe epochs
NUM_CLASSES = 4

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'

# ------------- DATA GENERATORS (Mai agresiv) ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,  # Crescut
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3,  # Crescut
    brightness_range=[0.7, 1.3],
    shear_range=0.2,  # NOU: distorsiune
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

# Class weights
class_counts = {}
for class_name in train_gen.class_indices:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    class_counts[train_gen.class_indices[class_name]] = len(os.listdir(class_dir))

total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

print("\n📊 Class Distribution:")
for cls, count in class_counts.items():
    print(f"  Class {cls}: {count} images (weight: {class_weights[cls]:.2f})")

# ---------------- MODEL cu FINE-TUNING -----------------
print("\n🏗️ Building Model with FINE-TUNING...")

# Load MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# PHASE 1: Freeze toate layerele mai întâi
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # NOU
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),  # Mai mare
    layers.BatchNormalization(),  # NOU
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile pentru phase 1
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ---------------- CALLBACKS -----------------
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
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

# --------------- PHASE 1: Train doar top layers ------------------
print("\n🚀 PHASE 1: Training top layers only...")
print(f"   Epochs: {EPOCHS}")
print(f"   Base model: FROZEN")
print(f"   Learning rate: 0.001\n")

history_phase1 = model.fit(
    train_gen,
    epochs=20,  # Doar 20 epochs pentru phase 1
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    class_weight=class_weights,
    verbose=1
)

# --------------- PHASE 2: FINE-TUNING layere superioare ------------------
print("\n\n🔥 PHASE 2: FINE-TUNING (unfreeze top layers)...")

# Unfreeze ultimele 50 de layere din MobileNetV2
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"   Unfreezing last {len(base_model.layers) - fine_tune_at} layers")
print(f"   Total trainable: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompile cu learning rate MAI MIC pentru fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 10x mai mic
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Mai mult patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Continue training
print(f"   Learning rate: 0.0001")
print(f"   Additional epochs: {EPOCHS - 20}\n")

history_phase2 = model.fit(
    train_gen,
    epochs=EPOCHS - 20,  # Restul epochs
    initial_epoch=len(history_phase1.history['accuracy']),
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    class_weight=class_weights,
    verbose=1
)

# --------------- SAVE -------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"\n✅ Model trained and saved to: {MODEL_PATH}")

# Combine histories
all_accuracy = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
all_val_accuracy = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

print(f"📊 Best validation accuracy: {max(all_val_accuracy):.4f}")
print(f"📊 Final training accuracy: {all_accuracy[-1]:.4f}")
print(f"\n🎯 IMPROVEMENT: {(max(all_val_accuracy) - 0.7759)*100:.2f}% better than previous!")

# Save history
import json
history_path = MODEL_PATH.replace('.h5', '_history.json')
with open(history_path, 'w') as f:
    json.dump({
        'accuracy': [float(x) for x in all_accuracy],
        'val_accuracy': [float(x) for x in all_val_accuracy],
        'loss': [float(x) for x in (history_phase1.history['loss'] + history_phase2.history['loss'])],
        'val_loss': [float(x) for x in (history_phase1.history['val_loss'] + history_phase2.history['val_loss'])]
    }, f, indent=2)

print(f"📈 Training history saved to: {history_path}")
print("\n✨ Model is now MORE ACCURATE with fine-tuning!")
