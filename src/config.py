"""
Global configuration for paths and directories.
"""

from pathlib import Path

# project_root/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# project_root/data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# project_root/data/processed
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
