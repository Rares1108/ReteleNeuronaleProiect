import os
import shutil
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed/"
TRAIN_DIR = "data/train/"
VAL_DIR = "data/validation/"
TEST_DIR = "data/test/"

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# ----------------------------------------------------
# Creează structurile train/validation/test
# ----------------------------------------------------
def prepare_split_folders(classes):
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for cls in classes:
            os.makedirs(os.path.join(folder, cls))

# ----------------------------------------------------
# Împarte fișierele stratificat
# ----------------------------------------------------
def split_data():
    classes = os.listdir(PROCESSED_DIR)
    prepare_split_folders(classes)

    for cls in classes:
        cls_path = os.path.join(PROCESSED_DIR, cls)
        images = os.listdir(cls_path)

        # generăm path complet pentru fiecare imagine
        image_paths = [os.path.join(cls_path, img) for img in images]

        # împărțire train vs temp
        train_imgs, temp_imgs = train_test_split(
            image_paths, 
            test_size=1 - SPLIT_RATIOS["train"], 
            shuffle=True
        )

        # împărțire temp → validation și test
        val_ratio = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=1 - val_ratio,
            shuffle=True
        )

        # copiere fișiere
        copy_images(train_imgs, TRAIN_DIR, cls)
        copy_images(val_imgs, VAL_DIR, cls)
        copy_images(test_imgs, TEST_DIR, cls)

        print(f"Clasa '{cls}' împărțită cu succes!")

    print("✔ Împărțirea în train/validation/test a fost finalizată!")


# ----------------------------------------------------
# Copiază imaginile în folderul țintă
# ----------------------------------------------------
def copy_images(image_list, target_dir, cls):
    for img_path in image_list:
        img_name = os.path.basename(img_path)
        dest = os.path.join(target_dir, cls, img_name)
        shutil.copy(img_path, dest)


# ----------------------------------------------------
# RUN SCRIPT
# ----------------------------------------------------
if __name__ == "__main__":
    print("Încep împărțirea setului processed/ în train/val/test...")
    split_data()
