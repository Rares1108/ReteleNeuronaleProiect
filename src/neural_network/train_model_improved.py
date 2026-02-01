import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_definition_improved import build_model_transfer_learning, build_model_improved_cnn

# ---------------- CONFIG ----------------
# IMPORTANT: Crește rezoluția pentru transfer learning!
IMG_SIZE = (128, 128)  # 128x128 vs 64x64 (mai multă informație)
BATCH_SIZE = 16  # Mai mic pentru model mai mare
EPOCHS = 30  # Mai multe epochs cu early stopping
NUM_CLASSES = 4

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = 'src/neural_network/saved_models/trained_model_improved.h5'

# ------------- DATA GENERATORS (Augmentation avansat) ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # Crescut de la 15
    width_shift_range=0.2,  # Crescut de la 0.1
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,  # NOU: zoom in/out
    brightness_range=[0.8, 1.2],  # NOU: variație lumină
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

# Calculare class weights pentru dezechilibru
class_counts = {}
for class_name in train_gen.class_indices:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    class_counts[train_gen.class_indices[class_name]] = len(os.listdir(class_dir))

total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

print("\n📊 Class Distribution:")
for cls, count in class_counts.items():
    print(f"  Class {cls}: {count} images (weight: {class_weights[cls]:.2f})")

# ---------------- MODEL -----------------
print("\n🏗️ Building Model with Transfer Learning...")
print("   Using MobileNetV2 pre-trained on ImageNet")

# OPȚIUNE 1: Transfer Learning (Recomandat - accuracy 75-85%)
model = build_model_transfer_learning(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES
)

# OPȚIUNE 2: CNN Îmbunătățit (Alternative - accuracy 70-75%)
# model = build_model_improved_cnn(
#     input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
#     num_classes=NUM_CLASSES
# )

model.summary()

# ---------------- CALLBACKS (Optimizări) -----------------
callbacks = [
    # Early Stopping: oprește antrenarea dacă val_accuracy nu crește
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Așteaptă 5 epochs fără îmbunătățire
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate când val_loss stagnează
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce lr cu 50%
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Salvare automată best model
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# --------------- TRAIN ------------------
print("\n🚀 Starting Training...")
print(f"   Epochs: {EPOCHS} (with early stopping)")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Image Size: {IMG_SIZE}")
print(f"   Augmentation: Advanced (rotation, zoom, brightness)")
print(f"   Class Weights: Enabled (balanced training)\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights,  # Balansare clase
    verbose=1
)

# --------------- SAVE -------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"\n✅ Model trained and saved to: {MODEL_PATH}")
print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# Salvare history pentru vizualizare
import json
history_path = MODEL_PATH.replace('.h5', '_history.json')
with open(history_path, 'w') as f:
    json.dump({
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }, f, indent=2)

print(f"📈 Training history saved to: {history_path}")
