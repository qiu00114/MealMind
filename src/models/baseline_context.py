"""
Simple context-aware baseline model for recommendations.
"""

import numpy as np
import pandas as pd


def build_context_stats(
    inter_fe: pd.DataFrame, rec_fe: pd.DataFrame
) -> pd.DataFrame:
    """
    Build simple context-aware statistics for (season, meal_type, recipe).

    Steps:
        - Filter rating to [1, 5].
        - Define `context` = season + "_" + meal_type.
        - Group by (context, recipe_id) to get avg_rating and n_interactions.
        - Define score = avg_rating * log1p(n_interactions).
        - Merge recipe features from rec_fe.

    Returns
    -------
    ctx_stats : pd.DataFrame
    """
    inter_cols_needed = [
        "user_id",
        "recipe_id",
        "rating",
        "season",
        "meal_type",
    ]
    missing_cols = [c for c in inter_cols_needed if c not in inter_fe.columns]
    if missing_cols:
        raise ValueError(
            f"inter_fe is missing required columns: {missing_cols}. "
            "Did you forget to call add_interaction_context_features()?"
        )

    ctx_df = inter_fe[inter_cols_needed].copy()
    ctx_df = ctx_df[(ctx_df["rating"] >= 1) & (ctx_df["rating"] <= 5)]

    ctx_df["context"] = ctx_df["season"].astype(str) + "_" + ctx_df["meal_type"].astype(
        str
    )

    ctx_stats = (
        ctx_df.groupby(["context", "recipe_id"])
        .agg(
            avg_rating=("rating", "mean"),
            n_interactions=("rating", "count"),
        )
        .reset_index()
    )

    ctx_stats["score"] = ctx_stats["avg_rating"] * np.log1p(ctx_stats["n_interactions"])

    # merge simple recipe features
    rec_fe_out = rec_fe[
        [
            "id",
            "name",
            "minutes",
            "n_steps",
            "n_ingredients",
            "quick_meal",
        ]
    ].copy()

    ctx_stats = ctx_stats.merge(
        rec_fe_out,
        left_on="recipe_id",
        right_on="id",
        how="left",
    )

    return ctx_stats


def recommend_for_context(
    ctx_stats: pd.DataFrame,
    season: str,
    meal_type: str,
    k: int = 10,
    min_interactions: int = 3,
) -> pd.DataFrame:
    """
    Recommend recipes for a given (season, meal_type) context.

    Parameters
    ----------
    ctx_stats : pd.DataFrame
        Output of build_context_stats().
    season : str
        One of 'winter', 'spring', 'summer', 'autumn'.
    meal_type : str
        One of 'breakfast', 'lunch', 'dinner', 'other'.
    k : int
        Number of recipes to return.
    min_interactions : int
        Minimum number of interactions to consider a recipe.

    Returns
    -------
    topk : pd.DataFrame
        DataFrame with columns:
            recipe_id, name, avg_rating, n_interactions, score,
            minutes, n_ingredients, quick_meal
        (if the columns exist in ctx_stats)
    """
    context_key = f"{season}_{meal_type}"
    subset = ctx_stats[ctx_stats["context"] == context_key].copy()
    subset = subset[subset["n_interactions"] >= min_interactions]
    subset = subset.sort_values("score", ascending=False)

    cols = [
        "recipe_id",
        "name",
        "avg_rating",
        "n_interactions",
        "score",
        "minutes",
        "n_ingredients",
        "quick_meal",
    ]
    cols = [c for c in cols if c in subset.columns]

    return subset.head(k)[cols]
