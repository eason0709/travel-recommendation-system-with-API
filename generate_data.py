"""Generate synthetic travel-behavior data with coherent preference clusters.

Root fix:
The previous synthetic data made some preferences too weak or entangled with
context variables. This version explicitly generates interpretable preference
profiles first, then derives context variables such as budget/weather/fatigue.
Clusters learned from selected_* features should therefore correspond to actual
preference types: nature, food, photo, culture, indoor, popular, etc.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data", exist_ok=True)

NUM_USERS = 20000

PREFERENCE_CENTERS = {
    # indoor, cost, photo, nature, culture, food, popularity
    "nature":   [0.7, 1.0, 2.6, 4.7, 0.9, 0.8, 1.8],
    "food":     [2.2, 2.3, 2.2, 0.6, 1.5, 4.7, 2.8],
    "photo":    [2.0, 2.1, 4.7, 1.3, 2.2, 1.8, 3.4],
    "culture":  [3.4, 1.8, 2.4, 0.8, 4.7, 1.1, 2.4],
    "popular":  [3.0, 3.0, 3.6, 1.0, 2.0, 2.4, 4.7],
    "indoor":   [4.7, 2.6, 2.1, 0.3, 2.8, 2.0, 2.5],
    "balanced": [2.6, 2.4, 2.6, 2.5, 2.5, 2.5, 2.6],
}

GROUP_NAMES = list(PREFERENCE_CENTERS)
GROUP_PROBS = np.array([0.16, 0.16, 0.15, 0.15, 0.14, 0.12, 0.12])
GROUP_PROBS = GROUP_PROBS / GROUP_PROBS.sum()


def clip_score(x):
    return float(np.clip(x, 0, 5))


def clip_01(x):
    return float(np.clip(x, 0, 1))


records = []
internal_group_counts = {name: 0 for name in GROUP_NAMES}

for user_id in range(1, NUM_USERS + 1):
    primary_group = np.random.choice(GROUP_NAMES, p=GROUP_PROBS)
    internal_group_counts[primary_group] += 1
    center = np.array(PREFERENCE_CENTERS[primary_group], dtype=float)

    # Some users are mixed, but the mixture is intentionally weaker than the
    # original version, so main preference types remain learnable.
    if np.random.rand() < 0.22:
        secondary_group = np.random.choice(GROUP_NAMES)
        second_center = np.array(PREFERENCE_CENTERS[secondary_group], dtype=float)
        mix_ratio = np.random.uniform(0.15, 0.35)
        center = center * (1 - mix_ratio) + second_center * mix_ratio

    prefs = np.clip(np.random.normal(center, 0.55), 0, 5)

    (
        selected_indoor_score,
        selected_cost_level,
        selected_photo_value,
        selected_nature_value,
        selected_culture_value,
        selected_food_value,
        selected_popularity,
    ) = [clip_score(v) for v in prefs]

    # Context variables are generated from preferences, but are not used to
    # train the cluster model. They remain useful for UI and later extensions.
    budget = np.clip(
        np.random.lognormal(mean=6.35 + selected_cost_level * 0.13, sigma=0.42),
        100,
        5000,
    )

    available_time = np.clip(
        np.random.normal(
            3.5
            + selected_nature_value * 0.45
            + selected_culture_value * 0.20
            - selected_food_value * 0.10,
            1.0,
        ),
        1,
        12,
    )

    weather_badness = clip_01(np.random.beta(2, 5) + np.random.normal(0, 0.06))

    social_context = clip_01(
        0.25
        + selected_food_value * 0.06
        + selected_popularity * 0.08
        + selected_photo_value * 0.04
        + np.random.normal(0, 0.13)
    )

    fatigue = clip_score(
        2.4
        + selected_indoor_score * 0.22
        - selected_nature_value * 0.18
        + np.random.normal(0, 0.75)
    )

    spontaneity = clip_score(
        2.2
        + selected_nature_value * 0.22
        + selected_popularity * 0.12
        - selected_indoor_score * 0.08
        + np.random.normal(0, 0.75)
    )

    # Small amount of noisy / inconsistent users.
    if np.random.rand() < 0.04:
        selected_indoor_score = clip_score(selected_indoor_score + np.random.normal(0, 1.2))
        selected_cost_level = clip_score(selected_cost_level + np.random.normal(0, 1.2))
        selected_photo_value = clip_score(selected_photo_value + np.random.normal(0, 1.2))
        selected_nature_value = clip_score(selected_nature_value + np.random.normal(0, 1.2))
        selected_culture_value = clip_score(selected_culture_value + np.random.normal(0, 1.2))
        selected_food_value = clip_score(selected_food_value + np.random.normal(0, 1.2))
        selected_popularity = clip_score(selected_popularity + np.random.normal(0, 1.2))

    records.append({
        "user_id": user_id,
        "budget": budget,
        "available_time": available_time,
        "weather_badness": weather_badness,
        "social_context": social_context,
        "fatigue": fatigue,
        "spontaneity": spontaneity,
        "selected_indoor_score": selected_indoor_score,
        "selected_cost_level": selected_cost_level,
        "selected_photo_value": selected_photo_value,
        "selected_nature_value": selected_nature_value,
        "selected_culture_value": selected_culture_value,
        "selected_food_value": selected_food_value,
        "selected_popularity": selected_popularity,
    })


df = pd.DataFrame(records)
df.to_csv("data/travel_behavior.csv", index=False, encoding="utf-8-sig")

print("Saved data/travel_behavior.csv")
print("Internal synthetic group counts for generation check:")
print(pd.Series(internal_group_counts).sort_values(ascending=False))
