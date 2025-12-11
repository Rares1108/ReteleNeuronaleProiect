# src/training/train_model.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

from src.neural_network.model_architecture import build_cnn

# Load config
CONFIG_PATH = "config/training_config.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

IMG_H = cfg["img_height"]
IMG_W = cfg["img_width"]
BATCH = cfg["batch_size"]
EPOCHS = cfg["epochs"]
TRAIN_DIR = cfg["train_dir"]
VAL_DIR = cfg["val_dir"]
TEST_DIR = cfg["test_dir"]
MODEL_PATH = cfg["model_path"]

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(cfg["plot_history_path"]), exist_ok=True)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# build model
num_classes = len(train_gen.class_indices)
model = build_cnn(input_shape=(IMG_H, IMG_W, 3), num_classes=num_classes, dropout_rate=0.4, lr=cfg["learning_rate"])
model.summary()

# callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early, reduce_lr]
)

# Save final model (best saved via checkpoint)
try:
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
except Exception as e:
    print("Error saving model:", e)

# Plot training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.tight_layout()
plt.savefig(cfg["plot_history_path"], dpi=200)
print("Saved training plot:", cfg["plot_history_path"])

# Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# load best model
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)

# predict
y_pred_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_gen.classes
class_indices = test_gen.class_indices
inv_map = {v:k for k,v in class_indices.items()}

# classification report
report = classification_report(y_true, y_pred, target_names=[inv_map[i] for i in range(len(inv_map))], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(cfg["classification_report_csv"], index=True)
print("Saved classification report to", cfg["classification_report_csv"])

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[inv_map[i] for i in range(len(inv_map))], yticklabels=[inv_map[i] for i in range(len(inv_map))])
plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
plt.savefig(cfg["confusion_matrix_path"], dpi=200)
print("Saved confusion matrix to", cfg["confusion_matrix_path"])

print("Test accuracy:", report['accuracy'])
