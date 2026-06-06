"""Train preference-based GMM clusters for the travel recommendation demo.

Root fix compared with the earlier version:
- Clustering is trained only on preference dimensions that can be directly
  matched to attraction features.
- Context variables such as budget, weather, fatigue and available_time are not
  used to define clusters. They can still be used later for filtering/penalty,
  but they should not decide whether a user is a nature/food/photo/culture type.
- This prevents cases where a nature-oriented input is assigned to a budget or
  cafe-like cluster just because context variables dominate the GMM.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Clustering should describe preference type, not budget/weather/context.
# These 7 features align one-to-one with attraction_features used by recommendation.
PREFERENCE_FEATURES = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

FEATURE_DISPLAY_NAMES = {
    "selected_indoor_score": "室內偏好",
    "selected_cost_level": "消費等級偏好",
    "selected_photo_value": "拍照價值偏好",
    "selected_nature_value": "自然景觀偏好",
    "selected_culture_value": "文化價值偏好",
    "selected_food_value": "美食價值偏好",
    "selected_popularity": "熱門程度偏好",
}

SHORT_NAMES = {
    "selected_indoor_score": "室內",
    "selected_cost_level": "高消費",
    "selected_photo_value": "拍照",
    "selected_nature_value": "自然",
    "selected_culture_value": "文化",
    "selected_food_value": "美食",
    "selected_popularity": "熱門",
}

STRONG_THRESHOLD = 0.80
MODERATE_THRESHOLD = 0.50
LOW_THRESHOLD = -0.50


def label_from_cluster_mean(row: pd.Series) -> str:
    """Conservative label from standardized cluster means."""
    row = row.astype(float)
    high = row[row >= STRONG_THRESHOLD].sort_values(ascending=False)
    moderate = row[row >= MODERATE_THRESHOLD].sort_values(ascending=False)
    low = row[row <= LOW_THRESHOLD].sort_values(ascending=True)

    if len(high) >= 2:
        names = [SHORT_NAMES.get(k, k) for k in high.index[:2]]
        return f"{names[0]}＋{names[1]}導向型"

    if len(high) == 1:
        key = high.index[0]
        return {
            "selected_indoor_score": "室內活動偏好型",
            "selected_cost_level": "高消費偏好型",
            "selected_photo_value": "拍照導向型",
            "selected_nature_value": "自然景點導向型",
            "selected_culture_value": "文化探索型",
            "selected_food_value": "美食導向型",
            "selected_popularity": "熱門景點導向型",
        }.get(key, f"{SHORT_NAMES.get(key, key)}導向型")

    if len(moderate) >= 2:
        names = [SHORT_NAMES.get(k, k) for k in moderate.index[:2]]
        return f"輕度{names[0]}＋{names[1]}型"

    if len(moderate) == 1:
        key = moderate.index[0]
        return f"輕度{SHORT_NAMES.get(key, key)}偏好型"

    if len(low) >= 2:
        names = [SHORT_NAMES.get(k, k) for k in low.index[:2]]
        return f"低{names[0]}低{names[1]}型"

    return "偏好不明顯型"


def label_basis_from_cluster_mean(row: pd.Series) -> str:
    row = row.astype(float)
    high = row[row >= MODERATE_THRESHOLD].sort_values(ascending=False)
    low = row[row <= LOW_THRESHOLD].sort_values(ascending=True)

    high_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(k, k)}={v:.2f}" for k, v in high.head(3).items()
    ) or "無明顯高於平均特徵"

    low_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(k, k)}={v:.2f}" for k, v in low.head(3).items()
    ) or "無明顯低於平均特徵"

    return f"命名依據：{high_text}；低特徵參考：{low_text}"


def description_from_cluster_mean(row: pd.Series) -> str:
    row = row.astype(float)
    high = row.sort_values(ascending=False).head(3)
    low = row.sort_values(ascending=True).head(3)
    high_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(k, k)}({v:.2f})" for k, v in high.items()
    )
    low_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(k, k)}({v:.2f})" for k, v in low.items()
    )
    return (
        "此 cluster 是根據使用者景點偏好特徵訓練而成，"
        "不使用預算、天氣、疲勞等情境欄位決定族群。"
        f"主要高於平均的偏好為：{high_text}。"
        f"主要低於平均的偏好為：{low_text}。"
    )


def build_cluster_label_summary(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, row in cluster_summary.iterrows():
        rows.append({
            "cluster": int(cluster_id),
            "cluster_label": label_from_cluster_mean(row),
            "label_basis": label_basis_from_cluster_mean(row),
            "description": description_from_cluster_mean(row),
            "top_high_features": "; ".join(f"{k}:{v:.3f}" for k, v in row.sort_values(ascending=False).head(5).items()),
            "top_low_features": "; ".join(f"{k}:{v:.3f}" for k, v in row.sort_values(ascending=True).head(5).items()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "travel_behavior.csv")

    missing = [col for col in PREFERENCE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"travel_behavior.csv 缺少必要欄位：{missing}")

    X = df[PREFERENCE_FEATURES]

    # Fit GMM on a representative sample for speed and stability, then predict all users.
    # This does not change the cluster definition: it is still preference-only.
    fit_df = df.sample(min(len(df), 5000), random_state=42)
    X_fit = fit_df[PREFERENCE_FEATURES]

    imputer = SimpleImputer(strategy="mean")
    X_fit_imputed = imputer.fit_transform(X_fit)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_imputed)

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    candidate_ks = range(2, 9)
    bic_scores = []

    for k in candidate_ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            n_init=2,
            max_iter=200,
            reg_covar=1e-6,
        )
        gmm.fit(X_fit_scaled)
        bic_scores.append(gmm.bic(X_fit_scaled))

    best_k = list(candidate_ks)[int(np.argmin(bic_scores))]

    print("\nBIC scores:")
    for k, bic in zip(candidate_ks, bic_scores):
        print(f"k={k}, BIC={bic:.2f}")
    print("\nBest k:", best_k)

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="diag",
        random_state=42,
        n_init=5,
        max_iter=300,
        reg_covar=1e-6,
    )
    gmm.fit(X_fit_scaled)

    soft_probs = gmm.predict_proba(X_scaled)
    hard_labels = np.argmax(soft_probs, axis=1)
    max_prob = soft_probs.max(axis=1)

    df["soft_cluster"] = hard_labels
    df["soft_cluster_confidence"] = max_prob

    for i in range(best_k):
        df[f"cluster_prob_{i}"] = soft_probs[:, i]

    sil_sample_size = min(len(X_scaled), 1000)
    sil_indices = np.random.default_rng(42).choice(len(X_scaled), size=sil_sample_size, replace=False)
    sil = silhouette_score(X_scaled[sil_indices], hard_labels[sil_indices])
    print("\nSilhouette Score:", sil)
    print("\nSoft Cluster Distribution:")
    print(df["soft_cluster"].value_counts().sort_index())
    print("\nCluster Confidence Summary:")
    print(df["soft_cluster_confidence"].describe())

    scaled_df = pd.DataFrame(X_scaled, columns=PREFERENCE_FEATURES)
    scaled_df["soft_cluster"] = hard_labels
    cluster_summary = scaled_df.groupby("soft_cluster")[PREFERENCE_FEATURES].mean()

    cluster_summary.to_csv(DATA_DIR / "soft_cluster_feature_mean.csv", encoding="utf-8-sig")

    label_summary = build_cluster_label_summary(cluster_summary)
    label_summary.to_csv(DATA_DIR / "soft_cluster_label_summary.csv", index=False, encoding="utf-8-sig")

    interpret_rows = []
    print("\nCluster Interpretation:\n")
    for cluster_id in cluster_summary.index:
        row = cluster_summary.loc[cluster_id]
        print(f"Cluster {cluster_id}: {label_from_cluster_mean(row)}")
        print("High features:")
        print(row.sort_values(ascending=False).head(5))
        print("Low features:")
        print(row.sort_values(ascending=True).head(5))
        print("-" * 60)

        for feature, value in row.sort_values(ascending=False).head(5).items():
            interpret_rows.append({
                "cluster": int(cluster_id),
                "direction": "high",
                "feature": feature,
                "z_mean": float(value),
            })
        for feature, value in row.sort_values(ascending=True).head(5).items():
            interpret_rows.append({
                "cluster": int(cluster_id),
                "direction": "low",
                "feature": feature,
                "z_mean": float(value),
            })

    pd.DataFrame(interpret_rows).to_csv(
        DATA_DIR / "soft_cluster_interpretation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(cluster_summary, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar(label="standardized cluster mean")
    plt.xticks(range(len(PREFERENCE_FEATURES)), PREFERENCE_FEATURES, rotation=35, ha="right")
    plt.yticks(range(len(cluster_summary.index)), cluster_summary.index)
    plt.title("Preference-based Soft Cluster Feature Mean Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "soft_cluster_feature_heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["soft_cluster_confidence"], bins=30)
    plt.title("Soft Cluster Confidence Distribution")
    plt.xlabel("Max Cluster Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "soft_cluster_confidence_distribution.png", dpi=300)
    plt.close()

    df.to_csv(DATA_DIR / "travel_behavior_soft_clustered.csv", index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "model_type": "preference_only_gmm",
            "features": PREFERENCE_FEATURES,
            "imputer": imputer,
            "scaler": scaler,
            "gmm": gmm,
            "cluster_summary": cluster_summary,
            "cluster_label_summary": label_summary,
            "note": "Clusters are trained only on preference features. Context variables are not used to define clusters.",
        },
        MODEL_DIR / "soft_cluster_pipeline.pkl",
    )

    print("\nSaved:")
    print(DATA_DIR / "travel_behavior_soft_clustered.csv")
    print(DATA_DIR / "soft_cluster_feature_mean.csv")
    print(DATA_DIR / "soft_cluster_label_summary.csv")
    print(DATA_DIR / "soft_cluster_interpretation.csv")
    print(MODEL_DIR / "soft_cluster_pipeline.pkl")


if __name__ == "__main__":
    main()
