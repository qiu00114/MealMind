"""
Simple context-aware baseline model for recommendations.

- Offline: build_context_stats(inter_fe, rec_fe) 生成 (season, meal_type, recipe) 统计表。
- Online:  recommend_for_now(...) / recommend_for_context(...) 做推荐。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data.features_interactions import month_to_season, guess_meal_type


def build_context_stats(
    inter_fe: pd.DataFrame,
    rec_fe: pd.DataFrame,
    min_rating: int = 1,
    max_rating: int = 5,
) -> pd.DataFrame:
    """
    Build simple context-aware statistics for (season, meal_type, recipe).

    Assumes inter_fe 至少有:
        - user_id
        - recipe_id
        - rating
        - season        (e.g., 'winter', 'spring', ...)
        - meal_type     (e.g., 'breakfast', 'lunch', 'dinner', 'late_night')

    rec_fe 至少有:
        - id
        - name
        - minutes
        - n_ingredients
        - quick_meal
    """
    inter = inter_fe.copy()

    # 过滤正常评分区间
    if "rating" in inter.columns:
        inter = inter[inter["rating"].between(min_rating, max_rating)]

    # 只保留需要的列
    needed_cols = ["season", "meal_type", "recipe_id", "rating"]
    for col in needed_cols:
        if col not in inter.columns:
            raise ValueError(f"inter_fe 缺少必须列: {col}")

    # (season, meal_type, recipe) 统计
    grouped = (
        inter.groupby(["season", "meal_type", "recipe_id"])["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_interactions", "mean": "avg_rating"})
    )

    # 一个非常简单的打分：评分越高、交互越多，score 越高
    grouped["score"] = grouped["avg_rating"] * np.log1p(grouped["n_interactions"])

    # 合并菜谱元数据
    rec_cols = ["id", "name", "minutes", "n_ingredients", "quick_meal"]
    rec_cols = [c for c in rec_cols if c in rec_fe.columns]
    rec_meta = rec_fe[rec_cols].rename(columns={"id": "recipe_id"})

    ctx_stats = grouped.merge(rec_meta, on="recipe_id", how="left")

    return ctx_stats


def load_context_stats(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load precomputed context_stats.csv.
    """
    if path is None:
        path = PROCESSED_DATA_DIR / "context_stats.csv"
    return pd.read_csv(path)


def recommend_for_context(
    ctx_stats: pd.DataFrame,
    season: str,
    meal_type: str,
    k: int = 10,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """
    在给定 season + meal_type 下，根据 score 排前 k 个菜。

    如果该情境下数据太少，就逐步后退到全局热门。
    """
    subset = ctx_stats.copy()

    # 先按 season + meal_type 过滤
    if "season" in subset.columns and "meal_type" in subset.columns:
        mask = (subset["season"] == season) & (subset["meal_type"] == meal_type)
        subset = subset[mask]

    # 控制交互次数下限
    if "n_interactions" in subset.columns and min_interactions is not None:
        subset = subset[subset["n_interactions"] >= min_interactions]

    # 如果过滤完为空，退一步：只看 season
    if subset.empty and "season" in ctx_stats.columns:
        subset = ctx_stats[ctx_stats["season"] == season].copy()
        if "n_interactions" in subset.columns and min_interactions is not None:
            subset = subset[subset["n_interactions"] >= min_interactions]

    # 再不行就用全局热门
    if subset.empty:
        subset = ctx_stats.copy()
        if "n_interactions" in subset.columns and min_interactions is not None:
            subset = subset[subset["n_interactions"] >= min_interactions]

    # 排序取前 k
    sort_col = "score" if "score" in subset.columns else "avg_rating"
    subset = subset.sort_values(sort_col, ascending=False)

    cols = [
        "recipe_id",
        "name",
        "avg_rating",
        "n_interactions",
        "score",
        "minutes",
        "n_ingredients",
        "quick_meal",
        "season",
        "meal_type",
    ]
    cols = [c for c in cols if c in subset.columns]

    return subset.head(k)[cols]


def recommend_for_now(
    ctx_stats: pd.DataFrame,
    when: Optional[pd.Timestamp] = None,
    k: int = 10,
    min_interactions: int = 5,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    根据当前时间 (或给定时间) 自动推断 season + meal_type 再推荐。
    """
    if when is None:
        when = pd.Timestamp.now(tz=tz)

    month = int(when.month)
    hour = int(when.hour)

    season = month_to_season(month)
    meal_type = guess_meal_type(hour)

    return recommend_for_context(
        ctx_stats=ctx_stats,
        season=season,
        meal_type=meal_type,
        k=k,
        min_interactions=min_interactions,
    )
