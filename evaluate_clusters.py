"""Evaluate recommendation methods against random baseline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from travel_utils import (
    ATTRACTION_FEATURES,
    BEHAVIOR_FEATURES,
    DATA_DIR,
    FIGURE_DIR,
    ensure_dirs,
    load_required_csv,
    matrix_cosine_similarity,
    sample_if_needed,
    save_csv,
)

DEBUG_MODE = False
MAX_USERS = 2_000 if DEBUG_MODE else None
TOP_K = 5
METHOD_SCORE_COLUMNS = {
    "personal_only": "personal_score",
    "cluster_only": "cluster_score",
    "collaborative_only": "collaborative_score",
    "hybrid": "hybrid_score",
}


def prepare_data():
    users = load_required_csv(DATA_DIR / "travel_behavior_soft_clustered.csv")
    attractions = load_required_csv(DATA_DIR / "attractions.csv")
    candidates = load_required_csv(DATA_DIR / "all_recommendation_candidates.csv")
    users = sample_if_needed(users, MAX_USERS)
    candidates = candidates[candidates["user_id"].isin(users["user_id"])].reset_index(drop=True)
    return users, attractions, candidates


def build_vector_maps(users: pd.DataFrame, attractions: pd.DataFrame):
    user_vectors = dict(zip(users["user_id"], users[BEHAVIOR_FEATURES].to_numpy(float)))
    attraction_vectors = dict(zip(attractions["attraction_id"], attractions[ATTRACTION_FEATURES].to_numpy(float)))
    return user_vectors, attraction_vectors


def row_similarity(row, user_vectors, attraction_vectors) -> float:
    return float(matrix_cosine_similarity(
        np.asarray(user_vectors[row["user_id"]]).reshape(1, -1),
        np.asarray(attraction_vectors[row["attraction_id"]]).reshape(1, -1),
    )[0, 0])


def random_baseline(users: pd.DataFrame, attractions: pd.DataFrame, user_vectors) -> float:
    rng = np.random.default_rng(42)
    attraction_matrix = attractions[ATTRACTION_FEATURES].to_numpy(float)
    scores = []
    for _, user in users.iterrows():
        item_pos = rng.integers(0, len(attractions))
        scores.append(matrix_cosine_similarity(user_vectors[user["user_id"]].reshape(1, -1), attraction_matrix[item_pos].reshape(1, -1))[0, 0])
    return float(np.mean(scores))


def evaluate_methods(users, attractions, candidates):
    user_vectors, attraction_vectors = build_vector_maps(users, attractions)
    baseline = random_baseline(users, attractions, user_vectors)
    rows = []

    for method_name, score_col in METHOD_SCORE_COLUMNS.items():
        top_recs = candidates.sort_values(["user_id", score_col], ascending=[True, False]).groupby("user_id").head(TOP_K)
        similarities = [row_similarity(row, user_vectors, attraction_vectors) for _, row in top_recs.iterrows()]
        mean_similarity = float(np.mean(similarities))
        rows.append({
            "method": method_name,
            "mean_similarity": mean_similarity,
            "random_baseline": baseline,
            "recommendation_lift": mean_similarity - baseline,
        })

    return pd.DataFrame(rows)


def save_comparison_plot(eval_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(eval_df["method"], eval_df["mean_similarity"])
    plt.axhline(eval_df["random_baseline"].iloc[0], linestyle="--", label="Random baseline")
    plt.ylabel("Mean Similarity")
    plt.title("Recommendation Method Comparison")
    plt.xticks(rotation=25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "recommendation_method_comparison.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs(DATA_DIR, FIGURE_DIR)
    users, attractions, candidates = prepare_data()
    eval_df = evaluate_methods(users, attractions, candidates)
    save_csv(eval_df, DATA_DIR / "recommendation_method_comparison.csv")
    save_comparison_plot(eval_df)

    best = eval_df.sort_values("mean_similarity", ascending=False).iloc[0]
    print("DEBUG MODE:", DEBUG_MODE)
    print("User count:", len(users))
    print("Candidate recommendation count:", len(candidates))
    print(eval_df)
    print("最佳方法:", best["method"])
    print("Saved: data/recommendation_method_comparison.csv")
    print("Saved: figures/recommendation_method_comparison.png")


if __name__ == "__main__":
    main()
