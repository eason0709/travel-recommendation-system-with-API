# 2

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/travel_behavior.csv")

features = [
    "budget",
    "available_time",
    "weather_badness",
    "social_context",
    "fatigue",
    "spontaneity",

    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

X = df[features]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =========================
# 用 BIC 自動選 GMM cluster 數
# =========================

candidate_ks = range(2, 9)

bic_scores = []

for k in candidate_ks:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

best_k = candidate_ks[int(np.argmin(bic_scores))]

print()
print("BIC scores:")
for k, bic in zip(candidate_ks, bic_scores):
    print(f"k={k}, BIC={bic:.2f}")

print()
print("Best k:")
print(best_k)

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

soft_probs = gmm.predict_proba(X_scaled)
hard_labels = np.argmax(soft_probs, axis=1)
max_prob = soft_probs.max(axis=1)

df["soft_cluster"] = hard_labels
df["soft_cluster_confidence"] = max_prob

for i in range(best_k):
    df[f"cluster_prob_{i}"] = soft_probs[:, i]

# =========================
# 評估 soft cluster
# =========================

sil = silhouette_score(X_scaled, hard_labels)

print()
print("Silhouette Score:")
print(sil)

print()
print("Soft Cluster Distribution:")
print(df["soft_cluster"].value_counts().sort_index())

print()
print("Cluster Confidence Summary:")
print(df["soft_cluster_confidence"].describe())

# =========================
# Standardized cluster mean
# =========================

scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df["soft_cluster"] = hard_labels

cluster_summary = scaled_df.groupby("soft_cluster")[features].mean()

cluster_summary.to_csv(
    "data/soft_cluster_feature_mean.csv",
    encoding="utf-8-sig"
)

# =========================
# 每群自動解釋
# =========================

interpret_rows = []

print()
print("Cluster Interpretation:")
print()

for cluster_id in cluster_summary.index:
    row = cluster_summary.loc[cluster_id]

    top_high = row.sort_values(ascending=False).head(5)
    top_low = row.sort_values(ascending=True).head(5)

    print(f"Cluster {cluster_id}")
    print("High features:")
    print(top_high)
    print()
    print("Low features:")
    print(top_low)
    print("-" * 60)

    for feature, value in top_high.items():
        interpret_rows.append({
            "cluster": cluster_id,
            "direction": "high",
            "feature": feature,
            "z_mean": value
        })

    for feature, value in top_low.items():
        interpret_rows.append({
            "cluster": cluster_id,
            "direction": "low",
            "feature": feature,
            "z_mean": value
        })

interpret_df = pd.DataFrame(interpret_rows)

interpret_df.to_csv(
    "data/soft_cluster_interpretation.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# Data-driven cluster labels
# =========================

FEATURE_DISPLAY_NAMES = {
    "budget": "預算",
    "available_time": "可用時間",
    "weather_badness": "天氣不佳程度",
    "social_context": "社交情境",
    "fatigue": "疲勞程度",
    "spontaneity": "隨興程度",
    "selected_indoor_score": "室內偏好",
    "selected_cost_level": "可接受消費",
    "selected_photo_value": "拍照價值",
    "selected_nature_value": "自然景點偏好",
    "selected_culture_value": "文化景點偏好",
    "selected_food_value": "美食偏好",
    "selected_popularity": "熱門程度偏好",
}


FEATURE_LABEL_NAMES = {
    "budget": "預算",
    "available_time": "長時間",
    "weather_badness": "壞天氣",
    "social_context": "社交",
    "fatigue": "低體力",
    "spontaneity": "隨興",
    "selected_indoor_score": "室內",
    "selected_cost_level": "高消費",
    "selected_photo_value": "拍照",
    "selected_nature_value": "自然",
    "selected_culture_value": "文化",
    "selected_food_value": "美食",
    "selected_popularity": "熱門",
}


PREFERENCE_FEATURES = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

CONTEXT_FEATURES = [
    "budget",
    "available_time",
    "weather_badness",
    "social_context",
    "fatigue",
    "spontaneity",
]

STRONG_THRESHOLD = 0.80
MODERATE_THRESHOLD = 0.50
EXTREME_BUDGET_THRESHOLD = 3.00
LOW_THRESHOLD = -0.50
STRONG_LOW_THRESHOLD = -0.80


def readable_feature(feature):
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def short_feature_name(feature):
    return FEATURE_LABEL_NAMES.get(feature, readable_feature(feature))


def _top_features(row, features, threshold):
    candidates = [
        (feature, float(row.get(feature, 0.0)))
        for feature in features
        if float(row.get(feature, 0.0)) >= threshold
    ]
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _low_features(row, features, threshold=LOW_THRESHOLD):
    candidates = [
        (feature, float(row.get(feature, 0.0)))
        for feature in features
        if float(row.get(feature, 0.0)) <= threshold
    ]
    return sorted(candidates, key=lambda item: item[1])


def infer_cluster_label(row):
    """Infer a readable cluster label from standardized cluster means.

    修正重點：
    - 不再用「最高特徵」直接命名，避免 budget=0.14 也被叫成預算偏高型。
    - 只有 z_mean >= 0.8 才視為明顯高特徵，可作為主要 label。
    - 若沒有明顯高特徵，但有明顯低特徵，改用「低 X 型」或「偏好不明顯型」。
    - 若所有特徵都接近平均，命名為「偏好不明顯型」。
    """

    row = row.astype(float)

    max_feature = row.idxmax()
    max_value = float(row.max())
    min_feature = row.idxmin()
    min_value = float(row.min())

    budget = float(row.get("budget", 0.0))
    weather = float(row.get("weather_badness", 0.0))
    indoor = float(row.get("selected_indoor_score", 0.0))
    cost = float(row.get("selected_cost_level", 0.0))
    photo = float(row.get("selected_photo_value", 0.0))
    nature = float(row.get("selected_nature_value", 0.0))
    culture = float(row.get("selected_culture_value", 0.0))
    food = float(row.get("selected_food_value", 0.0))
    popularity = float(row.get("selected_popularity", 0.0))

    # Extreme outlier handling.
    if budget >= EXTREME_BUDGET_THRESHOLD:
        return "極高預算特殊型"

    # If no feature is meaningfully high, do not force a high-feature label.
    if max_value < MODERATE_THRESHOLD:
        strong_lows = _low_features(row, PREFERENCE_FEATURES, threshold=STRONG_LOW_THRESHOLD)
        moderate_lows = _low_features(row, PREFERENCE_FEATURES, threshold=LOW_THRESHOLD)

        if len(strong_lows) >= 2:
            names = [short_feature_name(feature) for feature, _ in strong_lows[:2]]
            return f"低{names[0]}低{names[1]}型"

        if len(strong_lows) == 1:
            return f"低{short_feature_name(strong_lows[0][0])}型"

        if len(moderate_lows) >= 2:
            names = [short_feature_name(feature) for feature, _ in moderate_lows[:2]]
            return f"略低{names[0]}與{names[1]}型"

        return "偏好不明顯型"

    # Weather/context labels only when the signal is strong enough.
    if weather >= STRONG_THRESHOLD:
        if indoor >= MODERATE_THRESHOLD:
            return "壞天氣室內備案型"
        if culture >= MODERATE_THRESHOLD:
            return "壞天氣文化備案型"
        if food >= MODERATE_THRESHOLD:
            return "壞天氣美食備案型"
        return "壞天氣情境型"

    if weather <= STRONG_LOW_THRESHOLD:
        strong_preferences = _top_features(row, PREFERENCE_FEATURES, STRONG_THRESHOLD)
        if strong_preferences:
            return f"好天氣{short_feature_name(strong_preferences[0][0])}型"
        return "好天氣出遊型"

    # Budget labels require a clear budget signal.
    if budget >= STRONG_THRESHOLD:
        if cost >= MODERATE_THRESHOLD:
            return "高預算高消費型"
        return "高預算型"

    if budget <= STRONG_LOW_THRESHOLD:
        return "低預算型"

    # Preference-based labels.
    strong_preferences = _top_features(row, PREFERENCE_FEATURES, STRONG_THRESHOLD)

    if photo >= STRONG_THRESHOLD and popularity >= MODERATE_THRESHOLD:
        return "拍照打卡型"

    if popularity >= STRONG_THRESHOLD:
        if photo >= MODERATE_THRESHOLD or float(row.get("social_context", 0.0)) >= MODERATE_THRESHOLD:
            return "熱門社交打卡型"
        return "熱門景點導向型"

    if len(strong_preferences) >= 2:
        names = [short_feature_name(feature) for feature, _ in strong_preferences[:2]]
        return f"{names[0]}＋{names[1]}導向型"

    if len(strong_preferences) == 1:
        feature = strong_preferences[0][0]
        single_name_map = {
            "selected_indoor_score": "室內活動偏好型",
            "selected_cost_level": "高消費偏好型",
            "selected_photo_value": "拍照導向型",
            "selected_nature_value": "自然景點導向型",
            "selected_culture_value": "文化探索型",
            "selected_food_value": "美食導向型",
            "selected_popularity": "熱門景點導向型",
        }
        return single_name_map.get(feature, f"{short_feature_name(feature)}導向型")

    # Moderate labels: explicit that the signal is weak.
    moderate_preferences = _top_features(row, PREFERENCE_FEATURES, MODERATE_THRESHOLD)
    if len(moderate_preferences) >= 2:
        names = [short_feature_name(feature) for feature, _ in moderate_preferences[:2]]
        return f"輕度{names[0]}＋{names[1]}型"

    if len(moderate_preferences) == 1:
        return f"輕度{short_feature_name(moderate_preferences[0][0])}偏好型"

    # If only a context feature is moderately high, say it is weak/moderate.
    if max_feature in CONTEXT_FEATURES:
        return f"輕度{short_feature_name(max_feature)}情境型"

    return "偏好不明顯型"


def infer_cluster_label_strength(row):
    row = row.astype(float)
    max_abs = float(np.abs(row).max())
    max_value = float(row.max())

    if max_abs >= 1.0:
        return "strong"
    if max_abs >= 0.5:
        return "moderate"
    return "weak"


def infer_cluster_label_basis(row):
    row = row.astype(float)
    high = row[row >= MODERATE_THRESHOLD].sort_values(ascending=False)
    low = row[row <= LOW_THRESHOLD].sort_values(ascending=True)

    high_text = "、".join(
        f"{readable_feature(feature)}={value:.2f}"
        for feature, value in high.head(3).items()
    )
    low_text = "、".join(
        f"{readable_feature(feature)}={value:.2f}"
        for feature, value in low.head(3).items()
    )

    if not high_text:
        high_text = "無明顯高於平均特徵"
    if not low_text:
        low_text = "無明顯低於平均特徵"

    return f"命名依據：{high_text}；低特徵參考：{low_text}"


def infer_cluster_description(row):
    row = row.astype(float)
    high = row.sort_values(ascending=False).head(3)
    low = row.sort_values(ascending=True).head(3)

    high_text = "、".join(
        f"{readable_feature(feature)}({value:.2f})"
        for feature, value in high.items()
    )
    low_text = "、".join(
        f"{readable_feature(feature)}({value:.2f})"
        for feature, value in low.items()
    )

    return (
        "此標籤由目前 clustering 後的標準化特徵平均值自動推論。"
        "命名時只有 z_mean 達到門檻的特徵才會被用作主要標籤，"
        "避免將接近平均的微弱差異過度解讀。"
        f"主要高於平均的特徵為：{high_text}。"
        f"主要低於平均的特徵為：{low_text}。"
    )


def build_cluster_label_summary(cluster_summary):
    rows = []
    for cluster_id, row in cluster_summary.iterrows():
        rows.append({
            "cluster": int(cluster_id),
            "cluster_label": infer_cluster_label(row),
            "label_strength": infer_cluster_label_strength(row),
            "label_basis": infer_cluster_label_basis(row),
            "description": infer_cluster_description(row),
            "top_high_features": "; ".join(
                f"{feature}:{value:.3f}"
                for feature, value in row.sort_values(ascending=False).head(5).items()
            ),
            "top_low_features": "; ".join(
                f"{feature}:{value:.3f}"
                for feature, value in row.sort_values(ascending=True).head(5).items()
            ),
        })
    return pd.DataFrame(rows)


cluster_label_summary = build_cluster_label_summary(cluster_summary)
cluster_label_summary.to_csv(
    "data/soft_cluster_label_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("Data-driven Cluster Labels:")
print(cluster_label_summary[["cluster", "cluster_label"]])

# =========================
# Heatmap
# =========================

from matplotlib.colors import TwoSlopeNorm


def save_cluster_heatmap(cluster_summary, output_path):
    """
    Save an annotated red-blue heatmap for standardized cluster feature means.

    Red: feature mean is higher than the overall average.
    Blue: feature mean is lower than the overall average.
    Number in each cell: exact z_mean value rounded to 2 decimals.
    """

    values = cluster_summary.values.astype(float)

    # Use a symmetric range around zero so red/blue intensity is comparable.
    max_abs_value = max(
        2.0,
        float(np.nanmax(np.abs(values)))
    )

    norm = TwoSlopeNorm(
        vmin=-max_abs_value,
        vcenter=0.0,
        vmax=max_abs_value
    )

    fig_width = max(14, len(cluster_summary.columns) * 1.05)
    fig_height = max(6, len(cluster_summary.index) * 0.75)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized Mean Value (z_mean)")

    ax.set_xticks(range(len(cluster_summary.columns)))
    ax.set_xticklabels(
        cluster_summary.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(cluster_summary.index)))
    ax.set_yticklabels(cluster_summary.index)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Soft Cluster")
    ax.set_title("Soft Cluster Feature Mean Heatmap")

    # Draw cell boundaries for readability.
    ax.set_xticks(
        np.arange(-0.5, len(cluster_summary.columns), 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, len(cluster_summary.index), 1),
        minor=True
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.8
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add exact z_mean values inside cells.
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if abs(value) >= max_abs_value * 0.45 else "black"

            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_cluster_heatmap(
    cluster_summary=cluster_summary,
    output_path="figures/soft_cluster_feature_heatmap.png"
)

# =========================
# Confidence distribution
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df["soft_cluster_confidence"],
    bins=30
)

plt.title("Soft Cluster Confidence Distribution")
plt.xlabel("Max Cluster Probability")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "figures/soft_cluster_confidence_distribution.png",
    dpi=300
)

plt.show()

# =========================
# 儲存結果與模型
# =========================

df.to_csv(
    "data/travel_behavior_soft_clustered.csv",
    index=False,
    encoding="utf-8-sig"
)

joblib.dump(
    {
        "features": features,
        "imputer": imputer,
        "scaler": scaler,
        "gmm": gmm,
        "cluster_summary": cluster_summary,
        "cluster_label_summary": cluster_label_summary
    },
    "models/soft_cluster_pipeline.pkl"
)

print()
print("Saved:")
print("data/travel_behavior_soft_clustered.csv")
print("data/soft_cluster_feature_mean.csv")
print("data/soft_cluster_interpretation.csv")
print("data/soft_cluster_label_summary.csv")
print("models/soft_cluster_pipeline.pkl")