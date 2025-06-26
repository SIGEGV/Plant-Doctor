import os
import shutil
import random
from pathlib import Path

# Constants
SOURCE_DIR = Path.home() / "Desktop" / "Plant-Doctor" / "DATASET"
DEST_DIR = Path.home() / "Desktop" / "Plant-Doctor" / "Plant-Doctor-Test"
SPLIT_RATIO = 0.2 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_normal_folders(source_folder, dest_folder, ratio):
    for category in os.listdir(source_folder):
        category_path = os.path.join(source_folder, category)

        if not os.path.isdir(category_path) or category.lower() == "tomato":
            continue  

        for sub_class in os.listdir(category_path):
            sub_class_path = os.path.join(category_path, sub_class)
            if not os.path.isdir(sub_class_path):
                continue

            images = [f for f in os.listdir(sub_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)

            split_idx = int(len(images) * ratio)
            test_images = images[:split_idx]

            dest_subfolder = os.path.join(dest_folder, category, sub_class)
            create_dir(dest_subfolder)

            for img_name in test_images:
                src_path = os.path.join(sub_class_path, img_name)
                dest_path = os.path.join(dest_subfolder, img_name)

                shutil.copy2(src_path, dest_path)

            print(f"[{category}/{sub_class}] -> Copied {len(test_images)} test images.")

def copy_tomato_val(source_folder, dest_folder):
    val_dir = os.path.join(source_folder, "tomato", "val")
    dest_tomato = os.path.join(dest_folder, "tomato")

    if not os.path.exists(val_dir):
        print(" Tomato val directory not found!")
        return

    for sub_class in os.listdir(val_dir):
        src_sub_path = os.path.join(val_dir, sub_class)
        dest_sub_path = os.path.join(dest_tomato, sub_class)

        if os.path.isdir(src_sub_path):
            create_dir(dest_sub_path)

            for img in os.listdir(src_sub_path):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(src_sub_path, img), os.path.join(dest_sub_path, img))

            print(f"[tomato/{sub_class}] -> Copied {len(os.listdir(src_sub_path))} images from val.")


create_dir(DEST_DIR)

split_normal_folders(SOURCE_DIR, DEST_DIR, SPLIT_RATIO)


copy_tomato_val(SOURCE_DIR, DEST_DIR)

print("\n Dataset split complete. Test images saved to:", DEST_DIR)
