"""
Data loading and saving utilities for the Food.com dataset.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RAW_recipes.csv and RAW_interactions.csv from the given directory.

    Parameters
    ----------
    raw_data_dir : Path
        Directory that contains RAW_recipes.csv and RAW_interactions.csv.

    Returns
    -------
    recipes : pd.DataFrame
    interactions : pd.DataFrame
    """
    recipes_path = raw_data_dir / "RAW_recipes.csv"
    interactions_path = raw_data_dir / "RAW_interactions.csv"

    recipes = pd.read_csv(recipes_path)
    interactions = pd.read_csv(interactions_path)

    return recipes, interactions


def save_processed_data(
    inter_fe: pd.DataFrame,
    rec_fe: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[Path, Path]:
    """
    Save processed interactions and recipes to CSV files.

    Files:
        interactions_with_context.csv
        recipes_with_features.csv

    Returns
    -------
    inter_path : Path
    rec_path : Path
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    inter_fe_out = inter_fe[
        [
            "user_id",
            "recipe_id",
            "rating",
            "review",
            "date",
            "year",
            "month",
            "day_of_week",
            "is_weekend",
            "season",
            "hour",
            "meal_type",
        ]
    ].copy()

    rec_fe_out = rec_fe[
        [
            "id",
            "name",
            "minutes",
            "n_steps",
            "n_ingredients",
            "quick_meal",
            "tags_str",
            "ingredients_str",
            "steps_str",
            "text",
        ]
    ].copy()

    inter_path = processed_dir / "interactions_with_context.csv"
    rec_path = processed_dir / "recipes_with_features.csv"

    inter_fe_out.to_csv(inter_path, index=False)
    rec_fe_out.to_csv(rec_path, index=False)

    return inter_path, rec_path
