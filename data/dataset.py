import kagglehub
import os
import shutil
from pathlib import Path

# Create ./raw directory next to this file
CURRENT_DIR = Path(__file__).resolve().parent
RAW_DIR = CURRENT_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

# Download dataset
path = Path(kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions"))

# Move downloaded files into ./raw
for item in path.iterdir():
    if item.is_file():
        shutil.move(str(item), RAW_DIR)

print(f"Dataset downloaded and stored in: {RAW_DIR}")
