import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Path to folder with all the files
data_dir = Path("C:/Users/lab03/Documents/VesselGPTClean/Datos/Aneux+Intra-splines")
all_files = list(data_dir.glob("*.npy"))

# Separate by source
ane_files = [f for f in all_files if "-ane.npy" in f.name]
intra_files = [f for f in all_files if "-intra.npy" in f.name]

def split_files(files, seed=42):
    train_val, test = train_test_split(files, test_size=0.1, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.111, random_state=seed)
    return train, val, test

ane_train, ane_val, ane_test = split_files(ane_files)
intra_train, intra_val, intra_test = split_files(intra_files)

# Merge splits
train_files = ane_train + intra_train
val_files = ane_val + intra_val
test_files = ane_test + intra_test

# Shuffle final splits
random.seed(42)
random.shuffle(train_files)
random.shuffle(val_files)
random.shuffle(test_files)

# Save as text files (or copy if you prefer)
def save_split_absolute(file_list, split_name):
    with open(f"{split_name}_files.txt", "w") as f:
        for path in file_list:
            f.write(str(path.resolve()) + "\n")

def save_split_relative(file_list, split_name):
    with open(f"{split_name}_files.txt", "w") as f:
        for path in file_list:
            # Save relative to data_dir
            f.write(str(path.relative_to(data_dir)) + "\n")

save_split_relative(train_files, "splits/train")
save_split_relative(val_files, "splits/val")
save_split_relative(test_files, "splits/test")
