"""
Run a small end-to-end demo of the MealMind recommendation pipeline.

This script will:
1. Load the preprocessed data from PROCESSED_DATA_DIR.
2. Load the context-aware baseline statistics and print recommendations for "now".
3. Load the content-based model and print recommendations for a sample user.

Usage (from project root):

    python -m src.run_recommendations_demo
    python -m src.run_recommendations_demo --k 5
    python -m src.run_recommendations_demo --user-id 12345

"""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.models import (
    load_context_stats,
    recommend_for_now,
    load_default_content_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small demo of context-aware and content-based recommendations."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of recommendations to show for each method (default: 5).",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="User ID for the content-based recommender. "
             "If not provided, a random user from the interactions data will be used.",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="America/New_York",
        help="Timezone used when inferring 'now' for the context-aware baseline.",
    )
    return parser.parse_args()


def run_context_baseline_demo(k: int, tz: str = "America/New_York") -> None:
    print("\n==============================")
    print("  Context-aware baseline demo ")
    print("==============================")

    # Load precomputed context_stats.csv (built offline in the pipeline)
    try:
        ctx_stats = load_context_stats()
    except FileNotFoundError:
        path = PROCESSED_DATA_DIR / "context_stats.csv"
        print(f"[ERROR] Could not find context_stats.csv at: {path}")
        print("        Make sure you have run the preprocessing / pipeline step first.")
        return

    # Recommend for the current time
    recs_ctx = recommend_for_now(ctx_stats, k=k, tz=tz)

    print(f"\n[INFO] Top-{k} context-aware recommendations for 'now':\n")
    # Show a subset of columns if they exist
    cols_preferred = [
        "recipe_id",
        "name",
        "score",
        "avg_rating",
        "n_interactions",
        "season",
        "meal_type",
        "minutes",
        "n_ingredients",
        "quick_meal",
    ]
    cols = [c for c in cols_preferred if c in recs_ctx.columns]
    print(recs_ctx[cols].to_string(index=False))


def choose_user_id(inter_fe: pd.DataFrame, user_id: Optional[int]) -> Optional[int]:
    """Helper: choose a valid user_id from the interactions dataframe."""
    if inter_fe.empty:
        print("[ERROR] interactions data is empty.")
        return None

    if user_id is not None:
        # Check if this user_id actually exists in the data
        if user_id in inter_fe["user_id"].values:
            return user_id
        else:
            print(
                f"[WARN] Provided user_id={user_id} not found in interactions data. "
                "A random user will be selected instead."
            )

    # Fallback: pick a random user from the data
    return int(inter_fe["user_id"].sample(1, random_state=42).iloc[0])


def run_content_based_demo(k: int, user_id: Optional[int]) -> None:
    print("\n==============================")
    print("   Content-based model demo   ")
    print("==============================")

    try:
        model, inter_fe, rec_fe = load_default_content_model()
    except FileNotFoundError as e:
        print("[ERROR] Failed to load processed CSVs for the content-based model.")
        print(f"       Details: {e}")
        print("       Make sure you have run the preprocessing / pipeline step first.")
        return

    # Decide which user to use
    chosen_user_id = choose_user_id(inter_fe, user_id)
    if chosen_user_id is None:
        return

    print(f"\n[INFO] Using user_id={chosen_user_id} for content-based recommendations.")

    recs_cb = model.recommend_for_user(
        inter_fe,
        user_id=chosen_user_id,
        k=k,
        rating_col="rating",
        rating_threshold=4.0,
        exclude_rated=True,
    )

    if recs_cb is None or recs_cb.empty:
        print(
            "[INFO] No recommendations produced. "
            "This usually means the user does not have enough positive ratings."
        )
        return

    print(f"\n[INFO] Top-{k} content-based recommendations:\n")

    cols_preferred = [
        "recipe_id",
        "name",
        "sim",
        "minutes",
        "n_ingredients",
        "quick_meal",
    ]
    cols = [c for c in cols_preferred if c in recs_cb.columns]
    print(recs_cb[cols].to_string(index=False))


def main() -> None:
    args = parse_args()

    print("========================================")
    print("      MealMind Recommendation Demo      ")
    print("========================================")
    print(f"[INFO] PROCESSED_DATA_DIR = {PROCESSED_DATA_DIR}\n")

    # 1. Context-aware baseline
    run_context_baseline_demo(k=args.k, tz=args.tz)

    # 2. Content-based model
    run_content_based_demo(k=args.k, user_id=args.user_id)

    print("\n[INFO] Demo finished.\n")


if __name__ == "__main__":
    main()
