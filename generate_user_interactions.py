"""Generate simulated user-attraction interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from travel_utils import (
    ATTRACTION_FEATURES,
    BEHAVIOR_FEATURES,
    DATA_DIR,
    fill_numeric_na,
    load_required_csv,
    matrix_cosine_similarity,
    sample_if_needed,
    save_csv,
    softmax,
)

RNG = np.random.default_rng(42)
DEBUG_MODE = False
MAX_USERS = 2_000 if DEBUG_MODE else None
MAX_ATTRACTIONS = 50 if DEBUG_MODE else None
INTERACTIONS_PER_USER = 5


def attraction_penalties(user_row: pd.Series, attractions: pd.DataFrame) -> np.ndarray:
    time_penalty = np.maximum(0, attractions["estimated_time"].to_numpy() - user_row["available_time"]) * 0.08
    weather_penalty = user_row["weather_badness"] * np.maximum(0, 2.5 - attractions["indoor_score"].to_numpy()) * 0.10
    popularity_bias = attractions["popularity"].to_numpy() * 0.04
    return -time_penalty - weather_penalty + popularity_bias


def generate_interactions(users: pd.DataFrame, attractions: pd.DataFrame) -> pd.DataFrame:
    attraction_matrix = attractions[ATTRACTION_FEATURES].to_numpy(float)
    rows = []

    for idx, user in users.iterrows():
        if idx % 200 == 0:
            print(f"Processing user {idx} / {len(users)}")

        user_vec = user[BEHAVIOR_FEATURES].to_numpy(float).reshape(1, -1)
        similarities = matrix_cosine_similarity(user_vec, attraction_matrix).ravel()
        scores = similarities * 1.2 + attraction_penalties(user, attractions) + RNG.normal(0, 0.05, len(attractions))
        probabilities = softmax(scores, temperature=0.45)
        chosen_positions = RNG.choice(len(attractions), size=INTERACTIONS_PER_USER, replace=False, p=probabilities)

        chosen = attractions.iloc[chosen_positions]
        chosen_sims = similarities[chosen_positions]
        ratings = np.clip(
            2.5
            + chosen_sims * 2.0
            - np.maximum(0, chosen["estimated_time"].to_numpy() - user["available_time"]) * 0.15
            - user["weather_badness"] * np.maximum(0, 2.5 - chosen["indoor_score"].to_numpy()) * 0.15
            + RNG.normal(0, 0.45, len(chosen)),
            1,
            5,
        )

        rows.extend(
            {
                "user_id": user["user_id"],
                "soft_cluster": user["soft_cluster"],
                "attraction_id": attraction_id,
                "rating": rating,
                "clicked": int(rating >= 4.0),
            }
            for attraction_id, rating in zip(chosen["attraction_id"], ratings)
        )

    return pd.DataFrame(rows)


def main() -> None:
    users = fill_numeric_na(load_required_csv(DATA_DIR / "travel_behavior_soft_clustered.csv"))
    attractions = load_required_csv(DATA_DIR / "attractions.csv")
    users = sample_if_needed(users, MAX_USERS)
    attractions = sample_if_needed(attractions, MAX_ATTRACTIONS)

    interactions = generate_interactions(users, attractions)
    save_csv(interactions, DATA_DIR / "user_interactions.csv")

    print("data/user_interactions.csv 已產生")
    print("Dataset shape:", interactions.shape)
    print("Rating summary:")
    print(interactions["rating"].describe())
    print("Clicked distribution:")
    print(interactions["clicked"].value_counts())


if __name__ == "__main__":
    main()
