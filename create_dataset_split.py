import os
import random

random.seed(42)

base_dir = "/home/datasets/Breaking-Bad-Dataset.synthezised.bones/advancedTibiaHead/breaking_bad/bones"
bones_dir = "bones"
tibia_dirs = ["Tibia_R", "Tibia_L"]
train_file = os.path.join(os.path.dirname(base_dir), "bones.train.txt")
val_file = os.path.join(os.path.dirname(base_dir), "bones.val.txt")

all_paths = []
for tibia in tibia_dirs:
    tibia_path = os.path.join(base_dir, tibia)
    if os.path.exists(tibia_path):
        for folder in os.listdir(tibia_path):
            folder_path = os.path.join(bones_dir, tibia, folder)  # Relative path format
            if os.path.isdir(os.path.join(os.path.dirname(base_dir), folder_path)):  # Ensure it's a directory
                all_paths.append(folder_path)

random.shuffle(all_paths)
split_idx = int(0.8 * len(all_paths))
train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]

with open(train_file, "w") as f:
    for path in train_paths:
        f.write(f"{path}\n")

with open(val_file, "w") as f:
    for path in val_paths:
        f.write(f"{path}\n")

print(f"Train and val splits saved: {train_file}, {val_file}")
