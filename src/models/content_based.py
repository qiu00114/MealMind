"""
Content-based recommendation model using TF-IDF over recipe text.

- Uses the `text` column in recipes_with_features.csv (name + tags + ingredients + steps)
- Builds a user profile from the user's positively rated recipes
- Recommends similar recipes by cosine similarity
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROCESSED_DATA_DIR


class ContentBasedRecommender:
    """
    TF-IDF based content recommender.

    Parameters
    ----------
    rec_fe : pd.DataFrame
        Processed recipe dataframe, must contain:
        - id
        - text
        (optional: name, minutes, n_ingredients, quick_meal)
    text_col : str
        Name of the column with combined text features.
    id_col : str
        Name of the recipe id column.
    max_features : int
        Maximum number of TF-IDF features.
    """

    def __init__(
        self,
        rec_fe: pd.DataFrame,
        text_col: str = "text",
        id_col: str = "id",
        max_features: int = 50000,
    ) -> None:
        self.rec_fe = rec_fe.reset_index(drop=True).copy()
        self.text_col = text_col
        self.id_col = id_col

        if text_col not in self.rec_fe.columns:
            raise ValueError(f"rec_fe is missing required column: {text_col}")

        # 1) TF-IDF vectorization
        texts = self.rec_fe[text_col].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )
        self.recipe_tfidf = self.vectorizer.fit_transform(texts)

        # 2) Mapping from recipe_id -> row index in TF-IDF matrix
        self.recipe_id_to_idx = {
            rid: idx for idx, rid in enumerate(self.rec_fe[id_col].values)
        }

    # ---------- Internal: build user profile ----------

    def build_user_profile(
        self,
        inter_fe: pd.DataFrame,
        user_id: int | str,
        rating_col: str = "rating",
        rating_threshold: float = 4.0,
    ):
        """
        Build a user profile vector (1 x d) based on the user's
        positively rated recipes.

        inter_fe must contain:
            - user_id
            - recipe_id
            - rating
        """
        if rating_col not in inter_fe.columns:
            raise ValueError(f"inter_fe is missing rating column: {rating_col}")

        user_hist = inter_fe[
            (inter_fe["user_id"] == user_id)
            & (inter_fe[rating_col] >= rating_threshold)
        ]

        if user_hist.empty:
            return None

        # Find TF-IDF row indices for the positively rated recipes
        idxs = (
            user_hist["recipe_id"]
            .map(self.recipe_id_to_idx)
            .dropna()
            .astype(int)
            .tolist()
        )

        if not idxs:
            return None

        # Average TF-IDF vectors of those recipes
        user_profile = self.recipe_tfidf[idxs].mean(axis=0)
        return user_profile

    # ---------- Public API: recommend for a user ----------

    def recommend_for_user(
        self,
        inter_fe: pd.DataFrame,
        user_id: int | str,
        k: int = 10,
        rating_col: str = "rating",
        rating_threshold: float = 4.0,
        exclude_rated: bool = True,
    ) -> pd.DataFrame | None:
        """
        Simple content-based recommendation:

        1. Build a user profile (1 x d vector) from the user's
           positively rated recipes.
        2. Compute cosine similarity between the user profile and
           all recipe vectors.
        3. Return the Top-K most similar recipes.

        Returns
        -------
        pd.DataFrame or None
            Recommended recipes with similarity scores and metadata.
            Returns None if the user has no sufficient history.
        """
        user_profile = self.build_user_profile(
            inter_fe=inter_fe,
            user_id=user_id,
            rating_col=rating_col,
            rating_threshold=rating_threshold,
        )

        if user_profile is None:
            print(
                f"[INFO] User {user_id} does not have enough positively rated "
                f"history (rating >= {rating_threshold}). Cannot build a content-based profile."
            )
            return None

        # Ensure user_profile is a 2D numpy array (1 x d), not np.matrix
        import numpy as _np  # local import to avoid polluting global namespace
        user_profile = _np.asarray(user_profile)
        if user_profile.ndim == 1:
            user_profile = user_profile.reshape(1, -1)

        # Cosine similarity between user profile and all recipes
        sims = cosine_similarity(user_profile, self.recipe_tfidf).ravel()  # shape: (N,)


        rec_df = pd.DataFrame(
            {
                "recipe_idx": np.arange(len(sims)),
                "sim": sims,
            }
        )

        # Map row index -> recipe_id
        rec_df["recipe_id"] = rec_df["recipe_idx"].map(
            lambda idx: self.rec_fe.iloc[idx][self.id_col]
        )

        # Optionally exclude recipes that the user has already rated
        if exclude_rated and "recipe_id" in inter_fe.columns:
            user_hist_ids = set(
                inter_fe.loc[inter_fe["user_id"] == user_id, "recipe_id"].tolist()
            )
            rec_df = rec_df[~rec_df["recipe_id"].isin(user_hist_ids)]

        # Sort by similarity and keep Top-K
        rec_df = rec_df.sort_values("sim", ascending=False).head(k)

        # Merge with recipe metadata
        meta_cols = ["name", "minutes", "n_ingredients", "quick_meal"]
        meta_cols = [c for c in meta_cols if c in self.rec_fe.columns]

        result = rec_df.merge(
            self.rec_fe[[self.id_col] + meta_cols],
            left_on="recipe_id",
            right_on=self.id_col,
            how="left",
        )

        # Clean up redundant columns
        if self.id_col in result.columns:
            result = result.drop(columns=[self.id_col])
        if "recipe_idx" in result.columns:
            result = result.drop(columns=["recipe_idx"])

        return result


# ---------- Convenience: load from processed CSV ----------

def load_default_content_model() -> Tuple[ContentBasedRecommender, pd.DataFrame, pd.DataFrame]:
    """
    Load default processed data from PROCESSED_DATA_DIR:

        - recipes_with_features.csv
        - interactions_with_context.csv

    and construct a ContentBasedRecommender instance.

    Returns
    -------
    model : ContentBasedRecommender
        The content-based recommender initialized on recipe features.
    inter_fe : pd.DataFrame
        Processed interactions dataframe.
    rec_fe : pd.DataFrame
        Processed recipes dataframe.
    """
    rec_path = PROCESSED_DATA_DIR / "recipes_with_features.csv"
    inter_path = PROCESSED_DATA_DIR / "interactions_with_context.csv"

    rec_fe = pd.read_csv(rec_path)
    inter_fe = pd.read_csv(inter_path)

    model = ContentBasedRecommender(rec_fe)

    return model, inter_fe, rec_fe


if __name__ == "__main__":
    # Simple smoke test. In practice, you would call this from a notebook or script.
    model, inter_fe, rec_fe = load_default_content_model()
    example_user = inter_fe["user_id"].iloc[0]
    print(f"[DEBUG] Example user id: {example_user}")
    recs = model.recommend_for_user(inter_fe, user_id=example_user, k=5)
    print(recs)
