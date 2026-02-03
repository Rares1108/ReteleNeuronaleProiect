import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ---------------- CONFIG ----------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

TEST_DIR = "data/test"
MODEL_PATH = "src/neural_network/saved_models/trained_model.h5"
CLASS_NAMES = ["plastic", "hartie", "sticla", "metal"]

# ----------- LOAD MODEL -----------------
model = load_model(MODEL_PATH)

# ----------- LOAD TEST DATA -------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ----------- PREDICTIONS ----------------
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# ----------- METRICS + EXPORTS ----------
print("\n📊 Classification Report:\n")
report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(report_str)

out_dir = Path("docs/screenshots")
out_dir.mkdir(parents=True, exist_ok=True)

# Save classification report (string)
with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_str)

# Save classification report (JSON + Markdown table)
report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2)

md_lines = [
    "| Class | Precision | Recall | F1 | Support |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for cls in CLASS_NAMES:
    r = report_dict.get(cls, {})
    md_lines.append(
        f"| {cls} | {r.get('precision', 0):.2f} | {r.get('recall', 0):.2f} | {r.get('f1-score', 0):.2f} | {int(r.get('support', 0))} |"
    )
with open(out_dir / "classification_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# ----------- CONFUSION MATRIX (PNG) -----
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
print(f"\n🖼️ Saved confusion matrix to: {out_dir / 'confusion_matrix.png'}")
print(f"📝 Saved reports to: {out_dir / 'classification_report.txt'} and {out_dir / 'classification_report.md'}")
