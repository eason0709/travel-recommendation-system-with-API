"""Generate simulated travel behavior data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from travel_utils import DATA_DIR, clip_01, clip_score, ensure_dirs, save_csv

RNG = np.random.default_rng(42)
NUM_USERS = 20_000

GROUP_CENTERS = {
    "food": [1.6, 1.5, 2.2, 4.5, 3.0],
    "photo": [4.5, 2.3, 3.4, 2.3, 3.8],
    "nature": [2.0, 4.5, 1.8, 2.2, 2.0],
    "culture": [3.0, 2.0, 4.5, 2.4, 2.8],
    "trend": [4.0, 2.0, 3.0, 3.0, 4.6],
    "balanced": [2.8, 2.8, 2.8, 2.8, 2.8],
}
GROUP_NAMES = list(GROUP_CENTERS)
GROUP_PROBS = np.array([0.18, 0.16, 0.16, 0.16, 0.14, 0.20])
CENTER_MATRIX = np.array([GROUP_CENTERS[name] for name in GROUP_NAMES], dtype=float)


def generate_preference_centers(num_users: int) -> np.ndarray:
    primary_idx = RNG.choice(len(GROUP_NAMES), size=num_users, p=GROUP_PROBS)
    centers = CENTER_MATRIX[primary_idx].copy()

    mixed_mask = RNG.random(num_users) < 0.35
    secondary_idx = RNG.choice(len(GROUP_NAMES), size=mixed_mask.sum())
    mix_ratio = RNG.uniform(0.20, 0.45, size=mixed_mask.sum())[:, None]
    centers[mixed_mask] = centers[mixed_mask] * (1 - mix_ratio) + CENTER_MATRIX[secondary_idx] * mix_ratio
    return centers


def generate_users(num_users: int = NUM_USERS) -> pd.DataFrame:
    centers = generate_preference_centers(num_users)
    photo_pref, nature_pref, culture_pref, food_pref, popularity_pref = clip_score(
        RNG.normal(centers, 0.85)
    ).T

    adventure = (nature_pref - 2.5) / 1.5 + RNG.normal(0, 0.4, num_users)
    social = (food_pref + popularity_pref - 5) / 3 + RNG.normal(0, 0.4, num_users)
    luxury = RNG.normal(0, 1, num_users)
    comfort = RNG.normal(0, 1, num_users)
    trendiness = (popularity_pref - 2.5) / 1.5 + RNG.normal(0, 0.4, num_users)

    budget = np.clip(RNG.lognormal(mean=6.45 + luxury * 0.22, sigma=0.50), 100, 5000)
    fatigue = clip_score(2.5 + comfort * 0.35 - adventure * 0.15 + RNG.normal(0, 1.0, num_users))
    spontaneity = clip_score(2.5 + adventure * 0.45 + trendiness * 0.20 - comfort * 0.20 + RNG.normal(0, 0.95, num_users))
    available_time = np.clip(RNG.normal(5 + adventure * 0.35 + spontaneity * 0.15 - fatigue * 0.18, 1.6), 1, 12)
    weather_badness = clip_01(RNG.beta(2, 5, num_users) + RNG.normal(0, 0.06, num_users))
    social_context = clip_01(0.5 + social * 0.16 + RNG.normal(0, 0.18, num_users))

    fatigue_pressure = fatigue / 5
    budget_pressure = np.clip((850 - budget) / 850, 0, 1)
    spontaneity_level = spontaneity / 5

    selected_indoor_score = clip_score(2.0 + weather_badness * 0.75 + fatigue_pressure * 0.55 + comfort * 0.18 - adventure * 0.18 + culture_pref * 0.10 + RNG.normal(0, 1.05, num_users))
    selected_cost_level = clip_score(2.5 + luxury * 0.48 - budget_pressure * 0.55 + social_context * 0.22 + RNG.normal(0, 1.05, num_users))
    selected_photo_value = clip_score(1.3 + photo_pref * 0.48 + popularity_pref * 0.18 + social_context * 0.18 + spontaneity_level * 0.25 - fatigue_pressure * 0.18 + RNG.normal(0, 1.05, num_users))
    selected_nature_value = clip_score(1.3 + nature_pref * 0.50 + adventure * 0.20 - weather_badness * 0.50 - fatigue_pressure * 0.22 + spontaneity_level * 0.22 + RNG.normal(0, 1.05, num_users))
    selected_culture_value = clip_score(1.3 + culture_pref * 0.50 + photo_pref * 0.10 + selected_indoor_score * 0.08 + RNG.normal(0, 1.05, num_users))
    selected_food_value = clip_score(1.3 + food_pref * 0.50 + social_context * 0.30 + comfort * 0.12 + RNG.normal(0, 1.05, num_users))
    selected_popularity = clip_score(1.3 + popularity_pref * 0.50 + trendiness * 0.22 + social_context * 0.25 + photo_pref * 0.08 + RNG.normal(0, 1.05, num_users))

    preference_matrix = np.column_stack([
        selected_indoor_score,
        selected_cost_level,
        selected_photo_value,
        selected_nature_value,
        selected_culture_value,
        selected_food_value,
        selected_popularity,
    ])

    noisy_mask = RNG.random(num_users) < 0.12
    preference_matrix[noisy_mask] = clip_score(preference_matrix[noisy_mask] + RNG.normal(0, [1.2, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4], size=(noisy_mask.sum(), 7)))

    outlier_mask = RNG.random(num_users) < 0.025
    budget[outlier_mask] = RNG.choice([100, 5000], size=outlier_mask.sum())
    preference_matrix[outlier_mask] = RNG.uniform(0, 5, size=(outlier_mask.sum(), 7))

    return pd.DataFrame({
        "user_id": np.arange(1, num_users + 1),
        "budget": budget,
        "available_time": available_time,
        "weather_badness": weather_badness,
        "social_context": social_context,
        "fatigue": fatigue,
        "spontaneity": spontaneity,
        "selected_indoor_score": preference_matrix[:, 0],
        "selected_cost_level": preference_matrix[:, 1],
        "selected_photo_value": preference_matrix[:, 2],
        "selected_nature_value": preference_matrix[:, 3],
        "selected_culture_value": preference_matrix[:, 4],
        "selected_food_value": preference_matrix[:, 5],
        "selected_popularity": preference_matrix[:, 6],
    })


def main() -> None:
    ensure_dirs(DATA_DIR)
    df = generate_users()
    save_csv(df, DATA_DIR / "travel_behavior.csv")
    print("data/travel_behavior.csv 已產生")
    print("Dataset shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
