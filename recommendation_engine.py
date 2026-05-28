"""Score recommendation candidates and export Top-K recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from travel_utils import (
    ATTRACTION_FEATURES,
    BEHAVIOR_FEATURES,
    DATA_DIR,
    add_scene_names,
    fill_numeric_na,
    load_required_csv,
    matrix_cosine_similarity,
    sample_if_needed,
    save_csv,
)

DEBUG_MODE = False
MAX_USERS = 2_000 if DEBUG_MODE else None
MAX_ATTRACTIONS = 50 if DEBUG_MODE else None
TOP_K = 5


def prepare_data():
    users = fill_numeric_na(load_required_csv(DATA_DIR / "travel_behavior_soft_clustered.csv"))
    attractions = add_scene_names(load_required_csv(DATA_DIR / "attractions.csv"))
    interactions = load_required_csv(DATA_DIR / "user_interactions.csv")

    users = sample_if_needed(users, MAX_USERS)
    attractions = sample_if_needed(attractions, MAX_ATTRACTIONS)
    interactions = interactions[interactions["user_id"].isin(users["user_id"]) & interactions["attraction_id"].isin(attractions["attraction_id"])].reset_index(drop=True)
    return users, attractions, interactions


def build_rating_maps(interactions: pd.DataFrame):
    cluster_item_rating = interactions.groupby(["soft_cluster", "attraction_id"])["rating"].mean()
    global_item_rating = interactions.groupby("attraction_id")["rating"].mean()
    return cluster_item_rating.to_dict(), global_item_rating.to_dict()


def score_user_candidates(user: pd.Series, attractions: pd.DataFrame, cluster_behavior: pd.DataFrame, rating_maps) -> pd.DataFrame:
    cluster_rating_map, global_rating_map = rating_maps
    attraction_matrix = attractions[ATTRACTION_FEATURES].to_numpy(float)
    user_vec = user[BEHAVIOR_FEATURES].to_numpy(float).reshape(1, -1)
    cluster_vec = cluster_behavior.loc[user["soft_cluster"]].to_numpy(float).reshape(1, -1)

    personal = matrix_cosine_similarity(user_vec, attraction_matrix).ravel()
    personal -= np.maximum(0, attractions["estimated_time"].to_numpy() - user["available_time"]) * 0.08
    personal -= user["weather_badness"] * np.maximum(0, 2.5 - attractions["indoor_score"].to_numpy()) * 0.10

    cluster = matrix_cosine_similarity(cluster_vec, attraction_matrix).ravel()
    collaborative = np.array([
        (cluster_rating_map.get((user["soft_cluster"], attraction_id), global_rating_map.get(attraction_id, 3.0)) - 1) / 4
        for attraction_id in attractions["attraction_id"]
    ])

    confidence = float(user["soft_cluster_confidence"])
    weights = np.array([0.55, 0.20 * confidence, 0.25 * confidence])
    hybrid = (weights[0] * personal + weights[1] * cluster + weights[2] * collaborative) / weights.sum()

    result = attractions[["attraction_id", "name", "realistic_scene_name", "attraction_type"]].copy()
    result.insert(0, "user_id", user["user_id"])
    result.insert(1, "soft_cluster", user["soft_cluster"])
    result.insert(2, "cluster_confidence", confidence)
    result["personal_score"] = personal
    result["cluster_score"] = cluster
    result["collaborative_score"] = collaborative
    result["hybrid_score"] = hybrid
    result["recommendation_score"] = hybrid
    return result


def generate_recommendations(users: pd.DataFrame, attractions: pd.DataFrame, interactions: pd.DataFrame):
    cluster_behavior = users.groupby("soft_cluster")[BEHAVIOR_FEATURES].mean()
    rating_maps = build_rating_maps(interactions)
    seen = interactions.groupby("user_id")["attraction_id"].apply(set).to_dict()

    candidate_frames = []
    top_frames = []
    for idx, user in users.iterrows():
        if idx % 100 == 0:
            print(f"Processing user {idx} / {len(users)}")
        scored = score_user_candidates(user, attractions, cluster_behavior, rating_maps)
        user_seen = seen.get(user["user_id"], set())
        scored = scored[~scored["attraction_id"].isin(user_seen)]
        candidate_frames.append(scored)
        top_frames.append(scored.nlargest(TOP_K, "recommendation_score"))

    return pd.concat(candidate_frames, ignore_index=True), pd.concat(top_frames, ignore_index=True)


def main() -> None:
    users, attractions, interactions = prepare_data()
    print("DEBUG MODE:", DEBUG_MODE)
    print("User count:", len(users))
    print("Attraction count:", len(attractions))
    print("Interaction count:", len(interactions))

    all_candidates, recommendations = generate_recommendations(users, attractions, interactions)
    save_csv(all_candidates, DATA_DIR / "all_recommendation_candidates.csv")
    save_csv(recommendations, DATA_DIR / "recommendations.csv")
    print("data/all_recommendation_candidates.csv 已產生")
    print("data/recommendations.csv 已產生")
    print(recommendations.head(20))


if __name__ == "__main__":
    main()
