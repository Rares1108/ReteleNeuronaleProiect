import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_definition import build_model

# ---------------- CONFIG ----------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 4

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
MODEL_PATH = "src/neural_network/saved_models/trained_model.h5"

# ------------- DATA GENERATORS ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ---------------- MODEL -----------------
model = build_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES
)

model.summary()

# --------------- TRAIN ------------------
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# --------------- SAVE -------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

print("\n✅ Model antrenat și salvat la:", MODEL_PATH)
