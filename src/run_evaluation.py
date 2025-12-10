"""
Run offline evaluation for MealMind recommenders.

Models compared:
- Random baseline
- Context-aware baseline (global context statistics)
- Content-based TF-IDF model

Metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K
- Coverage
- Diversity (intra-list similarity)

Usage (from project root):

    python -m src.run_evaluation
    python -m src.run_evaluation --k 10 --max-users 500

"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DATA_DIR
from src.models import (
    load_context_stats,
    recommend_for_context,
    load_default_content_model,
)
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    coverage,
    diversity_intra_list_similarity,
)




# ---------------- Argument parsing ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for MealMind recommenders."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K cutoff for evaluation (default: 10).",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Ratings >= threshold are treated as positive interactions (default: 4.0).",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=2,
        help="Minimum number of positive interactions per user to be included (default: 2).",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=1000,
        help="Maximum number of users to evaluate (default: 1000). "
             "Set to a larger number if you want a more thorough evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------- Train/Test split ----------------


def train_test_split_by_user(
    interactions: pd.DataFrame,
    rating_col: str,
    rating_threshold: float,
    min_positive: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user:
      - consider only positive interactions (rating >= rating_threshold)
      - if user has at least `min_positive` positives, hold out 1 positive item as test
      - the rest of the user's interactions for that user go into train

    We return:
      - train_interactions: all interactions except the held-out positives
      - test_interactions: the held-out positives (one per user)
    """
    rng = np.random.default_rng(random_seed)

    train_parts = []
    test_parts = []

    # Work user by user; DO NOT drop from the full dataframe in each loop.
    interactions = interactions.copy()

    for user_id, user_df in interactions.groupby("user_id"):
        # positive interactions for this user
        user_pos = user_df[user_df[rating_col] >= rating_threshold]

        if len(user_pos) < min_positive:
            # not enough positives to form a train/test split
            continue

        # sample exactly one positive interaction as test for this user
        test_rows = user_pos.sample(n=1, random_state=random_seed)

        # the rest of THIS USER's interactions (both pos/neg) go to train
        train_rows = user_df.drop(index=test_rows.index)

        test_parts.append(test_rows)
        train_parts.append(train_rows)

    if not test_parts or not train_parts:
        raise RuntimeError(
            "After splitting, no train/test data found. "
            "Try lowering min_positive or rating_threshold."
        )

    train_inter = pd.concat(train_parts, ignore_index=True)
    test_inter = pd.concat(test_parts, ignore_index=True)

    return train_inter, test_inter


# ---------------- Main evaluation loop ----------------


def run_evaluation(args: argparse.Namespace) -> None:
    k = args.k
    rating_threshold = args.rating_threshold
    min_positive = args.min_positive
    max_users = args.max_users
    random_seed = args.random_seed

    print("========================================")
    print("        MealMind Offline Evaluation     ")
    print("========================================")
    print(f"[INFO] PROCESSED_DATA_DIR = {PROCESSED_DATA_DIR}")
    print(f"[INFO] K = {k}, rating_threshold = {rating_threshold}")
    print(f"[INFO] min_positive = {min_positive}, max_users = {max_users}\n")

    # 1) Load processed data
    inter_path = PROCESSED_DATA_DIR / "interactions_with_context.csv"
    rec_path = PROCESSED_DATA_DIR / "recipes_with_features.csv"

    interactions = pd.read_csv(inter_path)
    rec_fe = pd.read_csv(rec_path)

    print(f"[INFO] Loaded {len(interactions):,} interactions.")
    print(f"[INFO] Loaded {len(rec_fe):,} recipes.\n")

    all_item_ids = rec_fe["id"].unique().tolist()

    # 2) Train/test split
    print("[INFO] Splitting train/test by user...")
    train_inter, test_inter = train_test_split_by_user(
        interactions=interactions,
        rating_col="rating",
        rating_threshold=rating_threshold,
        min_positive=min_positive,
        random_seed=random_seed,
    )

    print(f"[INFO] Train interactions: {len(train_inter):,}")
    print(f"[INFO] Test interactions:  {len(test_inter):,}\n")

    # 3) Load models
    print("[INFO] Loading models...")
    # Context-aware baseline stats built offline from processed data
    ctx_stats = load_context_stats()

    # Content-based model + TF-IDF representation
    cb_model, _, _ = load_default_content_model()

    # For diversity: we re-use TF-IDF matrix from the content-based model
    item_id_to_idx = cb_model.recipe_id_to_idx
    item_vectors = cb_model.recipe_tfidf

    # 4) Prepare per-user test set
    # One positive held-out item per user
    user_test_items: Dict[int, List[int]] = (
        test_inter.groupby("user_id")["recipe_id"].apply(list).to_dict()
    )

    user_test_context = (
        test_inter.groupby("user_id")[["season", "meal_type"]].first().to_dict("index")
    )

    # Optionally subsample users for speed
    all_users = list(user_test_items.keys())
    if len(all_users) > max_users:
        rng = np.random.default_rng(random_seed)
        all_users = list(rng.choice(all_users, size=max_users, replace=False))

    print(f"[INFO] Evaluating on {len(all_users)} users.\n")

    # 5) Accumulators
    metrics_per_model = {
        "random": defaultdict(list),
        "context": defaultdict(list),
        "content": defaultdict(list),
    }

    all_recommended_random: List[List[int]] = []
    all_recommended_context: List[List[int]] = []
    all_recommended_content: List[List[int]] = []

    # 6) Evaluation loop
    for idx, user_id in enumerate(all_users, start=1):
        if idx % 100 == 0 or idx == 1:
            print(f"[INFO] Evaluating user {idx}/{len(all_users)} (user_id={user_id})")

        gt_items = set(user_test_items[user_id])  # usually size 1
        if not gt_items:
            continue

        # 6.1 Random baseline
        rng = np.random.default_rng(random_seed + user_id)
        random_recs = list(rng.choice(all_item_ids, size=k, replace=False))
        all_recommended_random.append(random_recs)

        metrics_per_model["random"]["precision"].append(
            precision_at_k(random_recs, gt_items, k)
        )
        metrics_per_model["random"]["recall"].append(
            recall_at_k(random_recs, gt_items, k)
        )
        metrics_per_model["random"]["hit_rate"].append(
            hit_rate_at_k(random_recs, gt_items, k)
        )
        metrics_per_model["random"]["ndcg"].append(
            ndcg_at_k(random_recs, gt_items, k)
        )
        metrics_per_model["random"]["diversity"].append(
            diversity_intra_list_similarity(random_recs, item_id_to_idx, item_vectors)
        )

        # 6.2 Context-aware baseline
        ctx = user_test_context[user_id]
        season = ctx.get("season", None)
        meal_type = ctx.get("meal_type", None)

        ctx_recs_df = recommend_for_context(
            ctx_stats=ctx_stats,
            season=season,
            meal_type=meal_type,
            k=k,
            min_interactions=5,
        )
        ctx_recs = ctx_recs_df["recipe_id"].tolist()
        all_recommended_context.append(ctx_recs)

        metrics_per_model["context"]["precision"].append(
            precision_at_k(ctx_recs, gt_items, k)
        )
        metrics_per_model["context"]["recall"].append(
            recall_at_k(ctx_recs, gt_items, k)
        )
        metrics_per_model["context"]["hit_rate"].append(
            hit_rate_at_k(ctx_recs, gt_items, k)
        )
        metrics_per_model["context"]["ndcg"].append(
            ndcg_at_k(ctx_recs, gt_items, k)
        )
        metrics_per_model["context"]["diversity"].append(
            diversity_intra_list_similarity(ctx_recs, item_id_to_idx, item_vectors)
        )

        # 6.3 Content-based TF-IDF model
        cb_recs_df = cb_model.recommend_for_user(
            train_inter,
            user_id=user_id,
            k=k,
            rating_col="rating",
            rating_threshold=rating_threshold,
            exclude_rated=True,
        )

        if cb_recs_df is None or cb_recs_df.empty:
            # user has no usable profile; skip for content-based
            continue

        cb_recs = cb_recs_df["recipe_id"].tolist()
        all_recommended_content.append(cb_recs)

        metrics_per_model["content"]["precision"].append(
            precision_at_k(cb_recs, gt_items, k)
        )
        metrics_per_model["content"]["recall"].append(
            recall_at_k(cb_recs, gt_items, k)
        )
        metrics_per_model["content"]["hit_rate"].append(
            hit_rate_at_k(cb_recs, gt_items, k)
        )
        metrics_per_model["content"]["ndcg"].append(
            ndcg_at_k(cb_recs, gt_items, k)
        )
        metrics_per_model["content"]["diversity"].append(
            diversity_intra_list_similarity(cb_recs, item_id_to_idx, item_vectors)
        )

    # 7) Aggregate metrics
    summary_rows = []
    all_model_names = ["random", "context", "content"]

    for model_name in all_model_names:
        m = metrics_per_model[model_name]

        if not m["precision"]:
            # This model might have been skipped for all users
            continue

        avg_precision = float(np.mean(m["precision"]))
        avg_recall = float(np.mean(m["recall"]))
        avg_hit = float(np.mean(m["hit_rate"]))
        avg_ndcg = float(np.mean(m["ndcg"]))
        avg_diversity = float(np.mean(m["diversity"]))

        if model_name == "random":
            cov = coverage(all_recommended_random, all_item_ids)
        elif model_name == "context":
            cov = coverage(all_recommended_context, all_item_ids)
        else:
            cov = coverage(all_recommended_content, all_item_ids)

        summary_rows.append(
            {
                "model": model_name,
                f"precision@{k}": avg_precision,
                f"recall@{k}": avg_recall,
                "hit_rate": avg_hit,
                f"ndcg@{k}": avg_ndcg,
                "coverage": cov,
                "diversity": avg_diversity,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("model")

    print("\n========== Evaluation Summary ==========\n")
    print(summary_df.round(4).to_string())
    print("\n========================================\n")

    # Save summary to CSV
    out_csv = PROCESSED_DATA_DIR / f"evaluation_summary_k{k}.csv"
    summary_df.to_csv(out_csv)
    print(f"[INFO] Saved summary to: {out_csv}")

    # 8) Plots
    make_plots(summary_df, k)


def make_plots(summary_df: pd.DataFrame, k: int) -> None:
    """
    Create a couple of simple comparison plots:
    - Precision@K bar chart
    - Coverage vs Diversity scatter plot
    """
    # Bar chart: Precision@K
    models = summary_df.index.tolist()
    precisions = summary_df[f"precision@{k}"].values

    plt.figure()
    plt.bar(models, precisions)
    plt.ylabel(f"Precision@{k}")
    plt.title(f"Precision@{k} Comparison")
    plt.tight_layout()
    out_png = PROCESSED_DATA_DIR / f"precision_at_{k}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"[INFO] Saved precision plot to: {out_png}")

    # Scatter plot: Coverage vs Diversity
    coverage_vals = summary_df["coverage"].values
    diversity_vals = summary_df["diversity"].values

    plt.figure()
    plt.scatter(coverage_vals, diversity_vals)

    for x, y, name in zip(coverage_vals, diversity_vals, models):
        plt.text(x, y, name, fontsize=9, ha="right", va="bottom")

    plt.xlabel("Coverage")
    plt.ylabel("Diversity")
    plt.title("Coverage vs Diversity")
    plt.tight_layout()
    out_png2 = PROCESSED_DATA_DIR / "coverage_vs_diversity.png"
    plt.savefig(out_png2)
    plt.close()
    print(f"[INFO] Saved coverage/diversity plot to: {out_png2}")


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
