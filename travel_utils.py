"""Shared utilities for the travel recommendation project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

BEHAVIOR_FEATURES = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

ATTRACTION_FEATURES = [
    "indoor_score",
    "cost_level",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity",
]

# Root fix: clusters should describe travel preference type, not context.
# Budget/weather/fatigue/time are still useful for recommendation penalties or UI,
# but they should not decide whether a user is nature/food/photo/culture type.
MODEL_FEATURES = [
    *BEHAVIOR_FEATURES,
]

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

PREFERENCE_FEATURES_FOR_LABEL = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

CONTEXT_FEATURES_FOR_LABEL = [
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


def _short_feature_name(feature: str) -> str:
    return FEATURE_LABEL_NAMES.get(feature, FEATURE_DISPLAY_NAMES.get(feature, feature))


def _top_features_for_label(row: pd.Series, features: list[str], threshold: float) -> list[tuple[str, float]]:
    candidates = [
        (feature, float(row.get(feature, 0.0)))
        for feature in features
        if float(row.get(feature, 0.0)) >= threshold
    ]
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _low_features_for_label(row: pd.Series, features: list[str], threshold: float = LOW_THRESHOLD) -> list[tuple[str, float]]:
    candidates = [
        (feature, float(row.get(feature, 0.0)))
        for feature in features
        if float(row.get(feature, 0.0)) <= threshold
    ]
    return sorted(candidates, key=lambda item: item[1])

# Fallback only. The current cluster labels should be generated from
# data/soft_cluster_label_summary.csv by cluster_interpretation_soft.py.
# This prevents fixed labels from conflicting with the heatmap after re-running GMM.
CLUSTER_LABELS = {}
CLUSTER_DESCRIPTIONS = {}


def load_cluster_label_summary(path: str | Path = DATA_DIR / "soft_cluster_label_summary.csv") -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def infer_cluster_label_from_row(row: pd.Series) -> str:
    """Infer a conservative cluster label from standardized cluster means.

    Only meaningful z-score differences are used for naming. This prevents
    labels such as "預算偏高型" when budget is only slightly above average.
    """
    row = row.astype(float)

    max_feature = row.idxmax()
    max_value = float(row.max())

    budget = float(row.get("budget", 0.0))
    weather = float(row.get("weather_badness", 0.0))
    indoor = float(row.get("selected_indoor_score", 0.0))
    cost = float(row.get("selected_cost_level", 0.0))
    photo = float(row.get("selected_photo_value", 0.0))
    culture = float(row.get("selected_culture_value", 0.0))
    food = float(row.get("selected_food_value", 0.0))
    popularity = float(row.get("selected_popularity", 0.0))

    if budget >= EXTREME_BUDGET_THRESHOLD:
        return "極高預算特殊型"

    if max_value < MODERATE_THRESHOLD:
        strong_lows = _low_features_for_label(row, PREFERENCE_FEATURES_FOR_LABEL, threshold=STRONG_LOW_THRESHOLD)
        moderate_lows = _low_features_for_label(row, PREFERENCE_FEATURES_FOR_LABEL, threshold=LOW_THRESHOLD)

        if len(strong_lows) >= 2:
            names = [_short_feature_name(feature) for feature, _ in strong_lows[:2]]
            return f"低{names[0]}低{names[1]}型"
        if len(strong_lows) == 1:
            return f"低{_short_feature_name(strong_lows[0][0])}型"
        if len(moderate_lows) >= 2:
            names = [_short_feature_name(feature) for feature, _ in moderate_lows[:2]]
            return f"略低{names[0]}與{names[1]}型"
        return "偏好不明顯型"

    if weather >= STRONG_THRESHOLD:
        if indoor >= MODERATE_THRESHOLD:
            return "壞天氣室內備案型"
        if culture >= MODERATE_THRESHOLD:
            return "壞天氣文化備案型"
        if food >= MODERATE_THRESHOLD:
            return "壞天氣美食備案型"
        return "壞天氣情境型"

    if weather <= STRONG_LOW_THRESHOLD:
        strong_preferences = _top_features_for_label(row, PREFERENCE_FEATURES_FOR_LABEL, STRONG_THRESHOLD)
        if strong_preferences:
            return f"好天氣{_short_feature_name(strong_preferences[0][0])}型"
        return "好天氣出遊型"

    if budget >= STRONG_THRESHOLD:
        if cost >= MODERATE_THRESHOLD:
            return "高預算高消費型"
        return "高預算型"

    if budget <= STRONG_LOW_THRESHOLD:
        return "低預算型"

    strong_preferences = _top_features_for_label(row, PREFERENCE_FEATURES_FOR_LABEL, STRONG_THRESHOLD)

    if photo >= STRONG_THRESHOLD and popularity >= MODERATE_THRESHOLD:
        return "拍照打卡型"

    if popularity >= STRONG_THRESHOLD:
        if photo >= MODERATE_THRESHOLD or float(row.get("social_context", 0.0)) >= MODERATE_THRESHOLD:
            return "熱門社交打卡型"
        return "熱門景點導向型"

    if len(strong_preferences) >= 2:
        names = [_short_feature_name(feature) for feature, _ in strong_preferences[:2]]
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
        return single_name_map.get(feature, f"{_short_feature_name(feature)}導向型")

    moderate_preferences = _top_features_for_label(row, PREFERENCE_FEATURES_FOR_LABEL, MODERATE_THRESHOLD)
    if len(moderate_preferences) >= 2:
        names = [_short_feature_name(feature) for feature, _ in moderate_preferences[:2]]
        return f"輕度{names[0]}＋{names[1]}型"

    if len(moderate_preferences) == 1:
        return f"輕度{_short_feature_name(moderate_preferences[0][0])}偏好型"

    if max_feature in CONTEXT_FEATURES_FOR_LABEL:
        return f"輕度{_short_feature_name(max_feature)}情境型"

    return "偏好不明顯型"


def infer_cluster_label_strength_from_row(row: pd.Series) -> str:
    row = row.astype(float)
    max_abs = float(np.abs(row).max())

    if max_abs >= 1.0:
        return "strong"
    if max_abs >= 0.5:
        return "moderate"
    return "weak"


def infer_cluster_label_basis_from_row(row: pd.Series) -> str:
    row = row.astype(float)
    high = row[row >= MODERATE_THRESHOLD].sort_values(ascending=False)
    low = row[row <= LOW_THRESHOLD].sort_values(ascending=True)

    high_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(feature, feature)}={value:.2f}"
        for feature, value in high.head(3).items()
    )
    low_text = "、".join(
        f"{FEATURE_DISPLAY_NAMES.get(feature, feature)}={value:.2f}"
        for feature, value in low.head(3).items()
    )

    if not high_text:
        high_text = "無明顯高於平均特徵"
    if not low_text:
        low_text = "無明顯低於平均特徵"

    return f"命名依據：{high_text}；低特徵參考：{low_text}"


def infer_cluster_description_from_row(row: pd.Series) -> str:
    row = row.astype(float)
    high = row.sort_values(ascending=False).head(3)
    low = row.sort_values(ascending=True).head(3)
    high_text = "、".join(f"{FEATURE_DISPLAY_NAMES.get(k, k)}({v:.2f})" for k, v in high.items())
    low_text = "、".join(f"{FEATURE_DISPLAY_NAMES.get(k, k)}({v:.2f})" for k, v in low.items())
    return (
        "此標籤由目前 clustering 後的標準化特徵平均值自動推論。"
        "命名時只有 z_mean 達到門檻的特徵才會被用作主要標籤，"
        "避免將接近平均的微弱差異過度解讀。"
        f"主要高於平均的特徵為：{high_text}。主要低於平均的特徵為：{low_text}。"
    )


def build_cluster_label_summary(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, row in cluster_summary.iterrows():
        rows.append({
            "cluster": int(cluster_id),
            "cluster_label": infer_cluster_label_from_row(row),
            "label_strength": infer_cluster_label_strength_from_row(row),
            "label_basis": infer_cluster_label_basis_from_row(row),
            "description": infer_cluster_description_from_row(row),
            "top_high_features": "; ".join(f"{k}:{v:.3f}" for k, v in row.sort_values(ascending=False).head(5).items()),
            "top_low_features": "; ".join(f"{k}:{v:.3f}" for k, v in row.sort_values(ascending=True).head(5).items()),
        })
    return pd.DataFrame(rows)


ATTRACTION_PROFILES = {
    "museum": [4.6, 2.8, 3.8, 0.5, 4.8, 1.2, 3.2, 2.5],
    "nature_trail": [0.4, 1.2, 4.0, 4.8, 0.8, 1.0, 2.8, 4.5],
    "night_market": [1.8, 2.0, 3.0, 0.3, 2.5, 4.8, 4.4, 2.2],
    "cafe_street": [3.3, 3.0, 4.3, 0.8, 2.5, 4.0, 3.8, 2.0],
    "historic_area": [2.2, 1.8, 3.8, 1.2, 4.5, 2.0, 3.2, 3.0],
    "shopping_mall": [4.8, 4.0, 2.8, 0.2, 1.8, 3.8, 4.3, 3.2],
    "park": [0.5, 0.8, 3.5, 4.2, 1.0, 1.3, 2.5, 2.8],
    "temple": [2.5, 1.0, 3.4, 1.2, 4.6, 1.5, 2.8, 2.0],
}

ATTRACTION_TYPES = list(ATTRACTION_PROFILES)
ATTRACTION_TYPE_PROBS = np.array([0.12, 0.13, 0.14, 0.14, 0.13, 0.12, 0.12, 0.10])
ATTRACTION_PROFILE_COLUMNS = [*ATTRACTION_FEATURES, "estimated_time"]

REALISTIC_SCENE_NAME_POOLS = {
    "museum": ["城市歷史博物館", "河岸美術館", "自然科學展示館", "地方文化博物館", "當代藝術展覽館"],
    "nature_trail": ["森林步道", "海岸觀景步道", "山區生態步道", "湖畔自然步道", "溪谷健行路線"],
    "night_market": ["在地夜市", "河岸觀光夜市", "老城小吃夜市", "車站商圈夜市", "廟口美食夜市"],
    "cafe_street": ["文青咖啡街", "老屋咖啡巷", "河岸咖啡街區", "文創甜點街", "巷弄咖啡聚落"],
    "historic_area": ["老街歷史街區", "古城文化街區", "傳統聚落保存區", "港町歷史街區", "紅磚建築街區"],
    "shopping_mall": ["大型購物中心", "車站百貨商圈", "室內娛樂商場", "複合式生活商場", "都會購物廣場"],
    "park": ["都會公園", "河濱公園", "森林公園", "親子休閒公園", "湖畔景觀公園"],
    "temple": ["百年古廟", "山城寺廟", "廟口文化廣場", "傳統信仰中心", "歷史寺院"],
}


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs or (DATA_DIR, FIGURE_DIR, MODEL_DIR):
        os.makedirs(directory, exist_ok=True)


def clip_score(x: Any) -> Any:
    return np.clip(x, 0, 5)


def clip_01(x: Any) -> Any:
    return np.clip(x, 0, 1)


def matrix_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = left @ right.T
    denom = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=float), where=denom != 0)


def vector_cosine_similarity(a: Any, b: Any) -> float:
    return float(matrix_cosine_similarity(np.asarray(a, dtype=float).reshape(1, -1), np.asarray(b, dtype=float).reshape(1, -1))[0, 0])


def softmax(scores: Any, temperature: float = 0.45) -> np.ndarray:
    values = np.asarray(scores, dtype=float) / temperature
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def fill_numeric_na(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
    return result


def sample_if_needed(df: pd.DataFrame, max_rows: int | None, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(max_rows, random_state=random_state).reset_index(drop=True)


def primary_attraction_type(attraction_type: Any) -> str:
    if pd.isna(attraction_type):
        return "unknown"
    return str(attraction_type).split("+")[0]


def build_realistic_scene_name(attraction_id: int, attraction_type: str) -> str:
    primary_type = primary_attraction_type(attraction_type)
    pool = REALISTIC_SCENE_NAME_POOLS.get(primary_type, ["旅遊景點"])
    base_name = pool[int(attraction_id) % len(pool)]
    suffix = f"複合景點 {int(attraction_id):03d}" if "+" in str(attraction_type) else f"{int(attraction_id):03d}"
    return f"{base_name} {suffix}"


def add_scene_names(attractions: pd.DataFrame) -> pd.DataFrame:
    result = attractions.copy()
    if "realistic_scene_name" not in result.columns:
        result["realistic_scene_name"] = [
            build_realistic_scene_name(row.attraction_id, row.attraction_type)
            for row in result.itertuples(index=False)
        ]
    return result


def cluster_label(cluster_id: int) -> str:
    cluster_id = int(cluster_id)
    label_summary = load_cluster_label_summary()
    if label_summary is not None and "cluster_label" in label_summary.columns:
        matched = label_summary[label_summary["cluster"].astype(int) == cluster_id]
        if len(matched) > 0:
            return str(matched.iloc[0]["cluster_label"])
    return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}（尚未產生資料驅動標籤）")


def cluster_description(cluster_id: int) -> str:
    cluster_id = int(cluster_id)
    label_summary = load_cluster_label_summary()
    if label_summary is not None and "description" in label_summary.columns:
        matched = label_summary[label_summary["cluster"].astype(int) == cluster_id]
        if len(matched) > 0:
            return str(matched.iloc[0]["description"])
    return CLUSTER_DESCRIPTIONS.get(cluster_id, "目前尚未產生此 cluster 的資料驅動語意解釋。請先執行 cluster_interpretation_soft.py。")


def cluster_title(cluster_id: int) -> str:
    return f"Cluster {int(cluster_id)}：{cluster_label(cluster_id)}"


def confidence_level(confidence: float) -> tuple[str, str]:
    confidence = float(confidence)
    if confidence >= 0.70:
        return "高信心", "模型對此使用者所屬 cluster 的判定較明確，可主要依據目前 cluster 解讀推薦結果。"
    if confidence >= 0.40:
        return "中等信心", "模型大致能判定使用者偏好，但仍建議參考 Soft Cluster Probability 中排名較高的其他 cluster。"
    return "低信心", "模型對單一 cluster 的判定不夠明確，表示使用者偏好可能分散於多個 cluster。建議不要只依賴目前 hard cluster 解釋。"


def load_required_csv(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到檔案：{path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
