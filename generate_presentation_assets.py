"""
Script pentru generarea automată a tuturor imaginilor necesare pentru PowerPoint
Generează:
1. Confusion Matrix (heatmap)
2. Bar chart metrici per clasă (Precision/Recall/F1)
3. Training history (accuracy & loss)
4. Dataset distribution bar chart
5. Sample images (4 exemple, câte unul per clasă)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import shutil
import random

# ==================== CONFIG ====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
TEST_DIR = "data/test"
MODEL_PATH = "src/neural_network/saved_models/trained_model_improved.h5"
HISTORY_PATH = "src/neural_network/saved_models/trained_model_improved_history.json"
CLASS_NAMES = ["hartie", "metal", "plastic", "sticla"]
CLASS_LABELS = ["Hârtie", "Metal", "Plastic", "Sticlă"]
OUTPUT_DIR = Path("docs/screenshots")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERARE IMAGINI PENTRU POWERPOINT")
print("=" * 60)

# ==================== 1. CONFUSION MATRIX ====================
print("\n📊 1/6 Generare Confusion Matrix...")

model = load_model(MODEL_PATH)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(test_gen, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            cbar_kws={'label': 'Număr predicții'})
plt.title("Matrice de Confuzie - Test Set", fontsize=14, fontweight='bold')
plt.ylabel("Clasă Reală", fontsize=12)
plt.xlabel("Clasă Prezisă", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvat: {OUTPUT_DIR / 'confusion_matrix.png'}")

# ==================== 2. METRICI PER CLASĂ (BAR CHART) ====================
print("\n📈 2/6 Generare Bar Chart Metrici...")

report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

metrics = ['precision', 'recall', 'f1-score']
metric_labels = ['Precizie', 'Recall', 'F1-Score']
x = np.arange(len(CLASS_LABELS))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [report_dict[cls][metric] for cls in CLASS_NAMES]
    ax.bar(x + i * width, values, width, label=label)

ax.set_xlabel('Clasă', fontsize=12)
ax.set_ylabel('Scor', fontsize=12)
ax.set_title('Metrici de Performanță per Clasă', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_LABELS)
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, metric in enumerate(metrics):
    values = [report_dict[cls][metric] for cls in CLASS_NAMES]
    for j, v in enumerate(values):
        ax.text(j + i * width, v + 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metrics_per_class.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvat: {OUTPUT_DIR / 'metrics_per_class.png'}")

# ==================== 3. TRAINING HISTORY ====================
print("\n📉 3/6 Generare Training History...")

if Path(HISTORY_PATH).exists():
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history['accuracy'], 'b-o', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Acuratețe Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Acuratețe', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, history['loss'], 'b-o', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax2.set_title('Loss Model', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvat: {OUTPUT_DIR / 'training_history.png'}")
else:
    print(f"   ⚠️ History file not found: {HISTORY_PATH}")
    print(f"   ℹ️ Skipping training history plot")

# ==================== 4. DATASET DISTRIBUTION ====================
print("\n📊 4/6 Generare Dataset Distribution...")

import os
from collections import defaultdict

splits = ["train", "validation", "test"]
data_dir = Path("data")

counts = {split: defaultdict(int) for split in splits}

for split in splits:
    for cls in CLASS_NAMES:
        cls_path = data_dir / split / cls
        if cls_path.exists():
            counts[split][cls] = sum(1 for f in cls_path.iterdir() if f.is_file())

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(CLASS_LABELS))
width = 0.25

for i, split in enumerate(splits):
    values = [counts[split][cls] for cls in CLASS_NAMES]
    ax.bar(x + i * width, values, width, label=split.capitalize())

ax.set_xlabel('Clasă', fontsize=12)
ax.set_ylabel('Număr Imagini', fontsize=12)
ax.set_title('Distribuția Dataset-ului', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_LABELS)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, split in enumerate(splits):
    values = [counts[split][cls] for cls in CLASS_NAMES]
    for j, v in enumerate(values):
        if v > 0:
            ax.text(j + i * width, v + 5, str(v), 
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dataset_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvat: {OUTPUT_DIR / 'dataset_distribution.png'}")

# ==================== 5. SAMPLE IMAGES ====================
print("\n🖼️ 5/6 Extragere Sample Images...")

sample_dir = OUTPUT_DIR / "samples"
sample_dir.mkdir(exist_ok=True)

for cls, label in zip(CLASS_NAMES, CLASS_LABELS):
    cls_path = data_dir / "test" / cls
    if cls_path.exists():
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
        if images:
            # Pick random image
            sample_img = random.choice(images)
            dest = sample_dir / f"sample_{cls}.jpg"
            shutil.copy(sample_img, dest)
            print(f"   ✅ Sample {label}: {dest}")

# Create composite image with all 4 samples
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Exemple din Dataset', fontsize=16, fontweight='bold')

for ax, cls, label in zip(axes, CLASS_NAMES, CLASS_LABELS):
    sample_path = sample_dir / f"sample_{cls}.jpg"
    if sample_path.exists():
        img = plt.imread(sample_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=14)
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dataset_samples.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvat: {OUTPUT_DIR / 'dataset_samples.png'}")

# ==================== 6. MODEL ARCHITECTURE DIAGRAM ====================
print("\n🏗️ 6/6 Generare Architecture Diagram...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Architecture layers (Transfer Learning - MobileNetV2)
layers = [
    ("Input\n128×128×3", 0.90),
    ("MobileNetV2\n(pretrained, frozen)", 0.75),
    ("GlobalAvgPool2D", 0.62),
    ("Dropout 0.5", 0.52),
    ("Dense (256)\nReLU", 0.40),
    ("Dropout 0.3", 0.30),
    ("Dense (4)\nSoftmax", 0.18),
]

colors = ['lightblue', 'lightgray', 'lightyellow', 'lightgreen', 'lightcoral', 
          'lightgreen', 'lightcoral']

for i, (layer, y) in enumerate(layers):
    # Draw box
    rect = plt.Rectangle((0.25, y - 0.05), 0.5, 0.10, 
                         facecolor=colors[i], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Add text
    ax.text(0.5, y, layer, ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Draw arrow (except for last layer)
    if i < len(layers) - 1:
        ax.arrow(0.5, y - 0.055, 0, -0.05, head_width=0.05, 
                head_length=0.015, fc='black', ec='black')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Arhitectura Transfer Learning (MobileNetV2)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_architecture.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvat: {OUTPUT_DIR / 'model_architecture.png'}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("✅ TOATE IMAGINILE AU FOST GENERATE CU SUCCES!")
print("=" * 60)
print(f"\n📁 Locație: {OUTPUT_DIR.absolute()}\n")
print("📋 Fișiere generate:")
print("   1. confusion_matrix.png - Matrice confuzie (heatmap)")
print("   2. metrics_per_class.png - Bar chart metrici per clasă")
print("   3. training_history.png - Accuracy & Loss pe epoch-uri")
print("   4. dataset_distribution.png - Distribuție imagini per split")
print("   5. dataset_samples.png - 4 exemple (câte unul per clasă)")
print("   6. model_architecture.png - Diagram arhitectură CNN")
print("\n💡 Folosește aceste imagini în PowerPoint!")
print("=" * 60)
