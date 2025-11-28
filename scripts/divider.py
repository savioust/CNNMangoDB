import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("data")
base_dir = data_dir

for split in ["train", "val", "test"]:
    (base_dir / split).mkdir(exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for class_dir in data_dir.iterdir():
    if class_dir.name in ["train", "val", "test"]:
        continue

    if class_dir.is_dir():
        print(f"📁 Dividindo classe: {class_dir.name}")
        images = list(class_dir.glob("*.*"))

        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest_dir = base_dir / split / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                dest_file = dest_dir / img.name
                if dest_file.exists():
                    new_name = dest_file.stem + "_copy" + dest_file.suffix
                    dest_file = dest_dir / new_name
                shutil.move(str(img), dest_file)

print("✅ Divisão concluída com sucesso!")
