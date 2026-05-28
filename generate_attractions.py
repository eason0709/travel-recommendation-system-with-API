"""Generate simulated attraction data with readable scene names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from travel_utils import (
    ATTRACTION_PROFILE_COLUMNS,
    ATTRACTION_PROFILES,
    ATTRACTION_TYPE_PROBS,
    ATTRACTION_TYPES,
    DATA_DIR,
    build_realistic_scene_name,
    clip_score,
    ensure_dirs,
    save_csv,
)

RNG = np.random.default_rng(42)
NUM_ATTRACTIONS = 300


def generate_attractions(num_attractions: int = NUM_ATTRACTIONS) -> pd.DataFrame:
    rows = []
    for attraction_id in range(1, num_attractions + 1):
        primary_type = RNG.choice(ATTRACTION_TYPES, p=ATTRACTION_TYPE_PROBS)
        base = np.array(ATTRACTION_PROFILES[primary_type], dtype=float)
        attraction_type = primary_type

        if RNG.random() < 0.30:
            secondary_type = RNG.choice(ATTRACTION_TYPES)
            mix = RNG.uniform(0.20, 0.45)
            base = base * (1 - mix) + np.array(ATTRACTION_PROFILES[secondary_type], dtype=float) * mix
            attraction_type = f"{primary_type}+{secondary_type}"

        feature_values = base.copy()
        feature_values[:7] = clip_score(feature_values[:7] + RNG.normal(0, [0.55, 0.65, 0.65, 0.65, 0.65, 0.65, 0.80]))
        feature_values[7] = float(np.clip(feature_values[7] + RNG.normal(0, 0.8), 0.8, 8.0))

        if RNG.random() < 0.08:
            feature_values[6] = clip_score(feature_values[6] + RNG.uniform(1.0, 2.0))
        if RNG.random() < 0.08:
            feature_values[6] = clip_score(feature_values[6] - RNG.uniform(1.0, 2.0))

        row = dict(zip(ATTRACTION_PROFILE_COLUMNS, feature_values))
        row.update({
            "attraction_id": attraction_id,
            "name": f"Attraction_{attraction_id:03d}_{primary_type}",
            "realistic_scene_name": build_realistic_scene_name(attraction_id, attraction_type),
            "attraction_type": attraction_type,
        })
        rows.append(row)

    columns = ["attraction_id", "name", "realistic_scene_name", "attraction_type", *ATTRACTION_PROFILE_COLUMNS]
    return pd.DataFrame(rows)[columns]


def main() -> None:
    ensure_dirs(DATA_DIR)
    df = generate_attractions()
    save_csv(df, DATA_DIR / "attractions.csv")
    print("data/attractions.csv 已產生")
    print("Dataset shape:", df.shape)
    print(df.head())
    print(df["attraction_type"].value_counts().head(20))


if __name__ == "__main__":
    main()
