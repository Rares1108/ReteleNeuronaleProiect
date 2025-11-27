import os
import cv2
import shutil

# ------------------------------
# CONFIG
# ------------------------------
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
IMG_SIZE = (64, 64)

# Mapping Kaggle -> Clase finale
CLASS_MAPPING = {
    "plastic": "plastic",
    "glass": "sticla",
    "metal": "metal",
    "paper": "hartie",
    "cardboard": "hartie",
    "trash": None  # Trash se elimina
}

# ------------------------------
# FUNCȚIE: pregătește directoarele
# ------------------------------
def prepare_folders():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    os.makedirs(PROCESSED_DIR)

    for cls in ["plastic", "hartie", "sticla", "metal"]:
        os.makedirs(os.path.join(PROCESSED_DIR, cls))

# ------------------------------
# FUNCȚIE: procesează o singură imagine
# ------------------------------
def process_image(path):
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        return img
    except Exception as e:
        print("EROARE imagine:", path, e)
        return None

# ------------------------------
# MAIN: Parcurge raw/ și salvează în processed/
# ------------------------------
def process_all():
    prepare_folders()

    for folder in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, folder)

        # dacă folderul nu este o clasă validă -> ignoră
        if folder not in CLASS_MAPPING:
            continue

        mapped_class = CLASS_MAPPING[folder]

        # trash -> eliminat
        if mapped_class is None:
            continue

        print(f"Procesez clasa: {folder} -> {mapped_class}")

        save_dir = os.path.join(PROCESSED_DIR, mapped_class)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            processed_img = process_image(file_path)
            if processed_img is None:
                continue

            # salvăm imaginea (ca jpg)
            save_name = f"{len(os.listdir(save_dir))}.jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

    print("✔ Procesare completă!")
    print(f"Imaginile rezultate se află în: {PROCESSED_DIR}")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    process_all()
