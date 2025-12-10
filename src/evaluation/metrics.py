# src/evaluation/metrics.py

"""
Evaluation metrics for recommender systems.

Includes:
- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K
- Coverage
- Diversity (intra-list similarity based on cosine similarity)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Set, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(
    recommended: Sequence[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Precision@K: fraction of recommended items in the Top-K that are relevant.
    """
    if k == 0:
        return 0.0

    recommended_k = list(recommended[:k])
    if not recommended_k:
        return 0.0

    hits = sum(1 for item in recommended_k if item in ground_truth)
    return hits / min(k, len(recommended_k))


def recall_at_k(
    recommended: Sequence[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Recall@K: fraction of relevant items that appear in the Top-K.
    """
    if not ground_truth:
        return 0.0

    recommended_k = list(recommended[:k])
    hits = sum(1 for item in recommended_k if item in ground_truth)
    return hits / len(ground_truth)


def hit_rate_at_k(
    recommended: Sequence[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Hit Rate@K: 1 if at least one relevant item appears in Top-K, else 0.
    """
    recommended_k = set(recommended[:k])
    if not ground_truth:
        return 0.0

    return 1.0 if recommended_k.intersection(ground_truth) else 0.0


def ndcg_at_k(
    recommended: Sequence[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    We assume binary relevance:
        rel = 1 if item in ground_truth, else 0.
    """
    recommended_k = list(recommended[:k])

    if not ground_truth or not recommended_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG (ideal DCG)
    max_rels = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, max_rels + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def coverage(
    all_recommended: Iterable[Sequence[int]],
    all_item_ids: Sequence[int],
) -> float:
    """
    Coverage: how many unique items are ever recommended,
    divided by the total number of items.
    """
    all_item_ids = list(all_item_ids)
    if not all_item_ids:
        return 0.0

    recommended_set = set()
    for rec_list in all_recommended:
        recommended_set.update(rec_list)

    return len(recommended_set) / len(all_item_ids)


def diversity_intra_list_similarity(
    recommended: Sequence[int],
    item_id_to_idx: Dict[int, int],
    item_vectors,
) -> float:
    """
    Diversity measure based on intra-list cosine similarity.

    We compute:
        avg_sim = average pairwise cosine similarity within the list
        diversity = 1 - avg_sim

    Parameters
    ----------
    recommended : Sequence[int]
        Sequence of recommended item IDs.
    item_id_to_idx : dict
        Mapping from item_id -> row index in item_vectors.
    item_vectors : array-like or sparse matrix, shape (n_items, n_features)
        Global item representations (e.g., TF-IDF matrix).

    Returns
    -------
    float
        Diversity score in [0, 1]. Higher is more diverse.
    """
    # Map IDs to indices in the vector matrix
    idxs = [item_id_to_idx[i] for i in recommended if i in item_id_to_idx]

    if len(idxs) <= 1:
        # List of size 0 or 1 is trivially "perfectly diverse"
        return 1.0

    submatrix = item_vectors[idxs]  # shape: (L, d)
    sim_matrix = cosine_similarity(submatrix)

    n = sim_matrix.shape[0]
    # Exclude diagonal
    sum_sim = sim_matrix.sum() - np.trace(sim_matrix)
    num_pairs = n * (n - 1)
    avg_sim = float(sum_sim / num_pairs)

    diversity = 1.0 - avg_sim
    # Clamp to [0, 1] just in case of numerical noise
    diversity = max(0.0, min(1.0, diversity))
    return diversity
