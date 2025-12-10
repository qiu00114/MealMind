import shutil
from pathlib import Path
import kagglehub


DATASET_ID = "shuyangli94/food-com-recipes-and-user-interactions"


def clear_dataset_cache():
    """
    Remove cache only for this specific dataset.
    This does NOT affect other KaggleHub cached datasets.
    """
    dataset_cache = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "shuyangli94"
        / "food-com-recipes-and-user-interactions"
    )

    if dataset_cache.exists():
        print(f"[INFO] Removing cached dataset:\n        {dataset_cache}")
        shutil.rmtree(dataset_cache, ignore_errors=True)
    else:
        print(f"[INFO] No existing cache found for:\n        {dataset_cache}")

    return dataset_cache


def download_dataset():
    """
    Download dataset via kagglehub and return the root directory
    where kagglehub stores it.
    """
    print(f"[INFO] Downloading dataset: {DATASET_ID}")
    root = kagglehub.dataset_download(DATASET_ID)
    root = Path(root)
    print(f"[INFO] KaggleHub dataset root:\n        {root}")
    return root


def move_files_to_raw(download_root: Path):
    """
    Move ALL files under the downloaded directory into ./raw.
    No extraction, no file type filtering.
    """
    script_dir = Path(__file__).resolve().parent       # e.g., MealMind/data
    raw_dir = script_dir / "raw"                      # e.g., MealMind/data/raw
    raw_dir.mkdir(exist_ok=True)

    print(f"[INFO] Target raw folder:\n        {raw_dir}")

    # Collect all files (any extension, any depth)
    files = [p for p in download_root.rglob("*") if p.is_file()]

    if not files:
        print("[WARN] No files found inside the downloaded dataset.")
        return raw_dir

    for f in files:
        target = raw_dir / f.name  # Flatten structure into raw/
        print(f"  [MOVE] {f} -> {target}")
        shutil.move(str(f), str(target))

    print("[INFO] All files have been moved to the raw folder.")
    return raw_dir


def main():
    print("=== MealMind Dataset Manager ===")

    # 1. Clear cache for THIS dataset only
    dataset_cache = clear_dataset_cache()
    print(f"[INFO] Dataset cache location:\n        {dataset_cache}")

    # 2. Download dataset again
    download_root = download_dataset()

    # 3. Move all dataset files to ./raw
    raw_dir = move_files_to_raw(download_root)

    print("\n=== Summary ===")
    print(f"  Dataset cache cleared:\n    {dataset_cache}")
    print(f"  Files stored in raw folder:\n    {raw_dir}")
    print("=== Done. ===")


if __name__ == "__main__":
    main()
