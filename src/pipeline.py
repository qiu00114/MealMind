"""
Full data preparation pipeline.
"""

from src.data_io import load_raw_data, save_processed_data
from src.features_interactions import clean_and_merge, add_interaction_context_features
from src.features_recipes import add_recipe_features
from src.baseline_context import build_context_stats
from src.config import PROCESSED_DATA_DIR


def run_full_pipeline():
    print("Running full MealMind data preparation pipeline...")

    # 1. Load raw data
    recipes, interactions = load_raw_data()

    # 2. Clean and merge
    rec_clean, inter_clean, inter_rec = clean_and_merge(recipes, interactions)

    # 3. Add interaction context features
    inter_fe = add_interaction_context_features(inter_rec)

    # 4. Add recipe content features
    rec_fe = add_recipe_features(rec_clean)

    # 5. Save processed outputs
    inter_path, rec_path = save_processed_data(inter_fe, rec_fe, PROCESSED_DATA_DIR)

    # 6. Baseline context-aware stats
    ctx_stats = build_context_stats(inter_fe, rec_fe)
    ctx_path = PROCESSED_DATA_DIR / "context_stats.csv"
    ctx_stats.to_csv(ctx_path, index=False)

    print("Done.")
    print(f"Processed interactions → {inter_path}")
    print(f"Processed recipes      → {rec_path}")
    print(f"Context stats          → {ctx_path}")


if __name__ == "__main__":
    run_full_pipeline()
