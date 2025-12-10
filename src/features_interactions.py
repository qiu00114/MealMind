"""
Feature engineering for user–recipe interactions.
"""

import pandas as pd


def month_to_season(m: int) -> str:
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def guess_meal_type(h) -> str:
    if pd.isna(h):
        return "other"
    h = int(h)
    if 5 <= h <= 10:
        return "breakfast"
    elif 11 <= h <= 15:
        return "lunch"
    elif 16 <= h <= 21:
        return "dinner"
    else:
        return "other"


def clean_and_merge(
    recipes: pd.DataFrame, interactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Basic cleaning of recipes and interactions, then merge them.

    - Drop interactions with missing rating.
    - Fill missing review with empty string.
    - Drop recipes missing name / ingredients / steps.
    - Drop duplicate recipes by id.
    - Merge interactions and recipes on recipe_id / id.

    Returns
    -------
    rec_clean : pd.DataFrame
    inter_clean : pd.DataFrame
    inter_rec : pd.DataFrame
        Merged interactions + recipes.
    """
    # interactions cleaning
    inter_clean = interactions.dropna(subset=["rating"]).copy()
    inter_clean["review"] = inter_clean["review"].fillna("")

    # recipes cleaning
    rec_clean = recipes.dropna(subset=["name", "ingredients", "steps"]).copy()
    rec_clean = rec_clean.drop_duplicates(subset=["id"])

    # merged
    inter_rec = inter_clean.merge(
        rec_clean,
        left_on="recipe_id",
        right_on="id",
        how="inner",
        suffixes=("_inter", "_rec"),
    )

    return rec_clean, inter_clean, inter_rec


def add_interaction_context_features(
    inter_rec: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add datetime-based context features to the merged interaction DataFrame.

    New columns:
        - date (as datetime)
        - year, month, day_of_week
        - is_weekend
        - season
        - hour
        - meal_type

    Returns
    -------
    inter_fe : pd.DataFrame
    """
    inter_fe = inter_rec.copy()
    inter_fe["date"] = pd.to_datetime(inter_fe["date"], errors="coerce")

    inter_fe["year"] = inter_fe["date"].dt.year
    inter_fe["month"] = inter_fe["date"].dt.month
    inter_fe["day_of_week"] = inter_fe["date"].dt.dayofweek
    inter_fe["is_weekend"] = inter_fe["day_of_week"].isin([5, 6]).astype(int)

    inter_fe["season"] = inter_fe["month"].apply(month_to_season)

    inter_fe["hour"] = inter_fe["date"].dt.hour
    inter_fe["meal_type"] = inter_fe["hour"].apply(guess_meal_type)

    return inter_fe
