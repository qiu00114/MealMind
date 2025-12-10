import shutil
from pathlib import Path
import kagglehub

from ..config import RAW_DATA_DIR

DATASET_ID = "shuyangli94/food-com-recipes-and-user-interactions"


def clear_dataset_cache():
    """
    Remove cache only for this specific dataset.
    """
    dataset_cache = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "shuyangli94"
        / "food-com-recipes-and-user-interactions"
    )
    shutil.rmtree(dataset_cache, ignore_errors=True)
    return dataset_cache


def download_dataset() -> Path:
    """
    Download dataset via kagglehub and return the root directory.
    """
    root = kagglehub.dataset_download(DATASET_ID)
    return Path(root)


def move_files_to_raw(download_root: Path) -> Path:
    """
    Move all files under the downloaded directory into ../data/raw.
    """
    raw_dir = RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    for f in download_root.rglob("*"):
        if f.is_file():
            target = raw_dir / f.name
            shutil.move(str(f), target)

    return raw_dir


def main():
    print("Setting up MealMind dataset...")

    cache_dir = clear_dataset_cache()
    download_root = download_dataset()
    raw_dir = move_files_to_raw(download_root)

    print(f"Cache cleared: {cache_dir}")
    print(f"Files stored in: {raw_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
