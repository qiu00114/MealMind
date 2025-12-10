"""
Feature engineering for recipe content.
"""

import pandas as pd


def _safe_to_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def add_recipe_features(rec_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple feature engineering for recipes:

    - Ensure `minutes`, `n_steps`, `n_ingredients` are numeric.
    - Add `quick_meal` (1 if minutes <= 30).
    - Create textual fields: tags_str, ingredients_str, steps_str, text.

    Returns
    -------
    rec_fe : pd.DataFrame
    """
    rec_fe = rec_clean.copy()

    for col in ["minutes", "n_steps", "n_ingredients"]:
        rec_fe[col] = pd.to_numeric(rec_fe[col], errors="coerce")

    rec_fe["quick_meal"] = (rec_fe["minutes"] <= 30).astype(int)

    rec_fe["ingredients_str"] = rec_fe["ingredients"].apply(_safe_to_str)
    rec_fe["tags_str"] = rec_fe["tags"].apply(_safe_to_str)
    rec_fe["steps_str"] = rec_fe["steps"].apply(_safe_to_str)

    rec_fe["text"] = (
        rec_fe["name"].astype(str)
        + " "
        + rec_fe["tags_str"]
        + " "
        + rec_fe["ingredients_str"]
        + " "
        + rec_fe["steps_str"]
    )

    return rec_fe
