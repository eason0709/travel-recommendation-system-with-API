"""
test_cluster_inputs.py

用途：
用來快速測試不同使用者輸入參數下，GMM soft clustering 會輸出哪個 cluster、
confidence 多少，以及各 cluster probability。

放置位置：
請把本檔案放在專案根目錄，也就是和 app.py、cluster_interpretation_soft.py 同一層。

執行方式：
1. 測試內建案例：
   python test_cluster_inputs.py --preset

2. 測試單一自訂輸入：
   python test_cluster_inputs.py --budget 3 --available-time 4 --weather-badness 0.2 --social-context 0.5 --fatigue 2 --spontaneity 3 --indoor 4 --cost 2 --photo 5 --nature 1 --culture 4 --food 3 --popularity 5

3. 顯示完整 probability：
   python test_cluster_inputs.py --preset --show-prob

必要檔案：
models/soft_cluster_pipeline.pkl
data/soft_cluster_label_summary.csv  可選，若存在會顯示中文 cluster label
data/soft_cluster_feature_mean.csv   可選，若存在會顯示 cluster 主要高低特徵
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES = [
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


DISPLAY_NAMES = {
    "budget": "預算 budget",
    "available_time": "可用時間 available_time",
    "weather_badness": "天氣不佳程度 weather_badness",
    "social_context": "同行/社交情境 social_context",
    "fatigue": "疲勞程度 fatigue",
    "spontaneity": "隨興程度 spontaneity",
    "selected_indoor_score": "室內偏好 indoor",
    "selected_cost_level": "消費等級 cost",
    "selected_photo_value": "拍照價值 photo",
    "selected_nature_value": "自然景觀 nature",
    "selected_culture_value": "文化價值 culture",
    "selected_food_value": "美食價值 food",
    "selected_popularity": "熱門程度 popularity",
}


PRESET_CASES = [
    {
        "case_name": "低預算_室內文化型",
        "budget": 1,
        "available_time": 3,
        "weather_badness": 0.8,
        "social_context": 0.3,
        "fatigue": 4,
        "spontaneity": 1,
        "selected_indoor_score": 5,
        "selected_cost_level": 1,
        "selected_photo_value": 2,
        "selected_nature_value": 1,
        "selected_culture_value": 5,
        "selected_food_value": 2,
        "selected_popularity": 2,
    },
    {
        "case_name": "高預算_熱門打卡型",
        "budget": 5,
        "available_time": 4,
        "weather_badness": 0.2,
        "social_context": 1.0,
        "fatigue": 1,
        "spontaneity": 4,
        "selected_indoor_score": 3,
        "selected_cost_level": 5,
        "selected_photo_value": 5,
        "selected_nature_value": 2,
        "selected_culture_value": 3,
        "selected_food_value": 4,
        "selected_popularity": 5,
    },
    {
        "case_name": "自然景點導向型",
        "budget": 3,
        "available_time": 5,
        "weather_badness": 0.0,
        "social_context": 0.3,
        "fatigue": 1,
        "spontaneity": 3,
        "selected_indoor_score": 1,
        "selected_cost_level": 2,
        "selected_photo_value": 4,
        "selected_nature_value": 5,
        "selected_culture_value": 1,
        "selected_food_value": 2,
        "selected_popularity": 2,
    },
    {
        "case_name": "美食消費導向型",
        "budget": 4,
        "available_time": 3,
        "weather_badness": 0.4,
        "social_context": 0.8,
        "fatigue": 2,
        "spontaneity": 4,
        "selected_indoor_score": 4,
        "selected_cost_level": 4,
        "selected_photo_value": 3,
        "selected_nature_value": 1,
        "selected_culture_value": 2,
        "selected_food_value": 5,
        "selected_popularity": 4,
    },
    {
        "case_name": "中性模糊型",
        "budget": 3,
        "available_time": 3,
        "weather_badness": 0.5,
        "social_context": 0.5,
        "fatigue": 2.5,
        "spontaneity": 2.5,
        "selected_indoor_score": 2.5,
        "selected_cost_level": 2.5,
        "selected_photo_value": 2.5,
        "selected_nature_value": 2.5,
        "selected_culture_value": 2.5,
        "selected_food_value": 2.5,
        "selected_popularity": 2.5,
    },
]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_model_bundle(path: Path) -> dict[str, Any]:
    obj = load_pickle(path)

    # Most recommended format:
    # {"imputer": ..., "scaler": ..., "gmm": ..., "features": [...]}
    if isinstance(obj, dict):
        return obj

    # If the file is a sklearn Pipeline, keep it as pipeline.
    return {"pipeline": obj, "features": FEATURES}


def get_features(bundle: dict[str, Any]) -> list[str]:
    for key in ["features", "feature_names", "input_features"]:
        if key in bundle and bundle[key] is not None:
            return list(bundle[key])
    return FEATURES


def predict_cluster(bundle: dict[str, Any], input_df: pd.DataFrame) -> tuple[int, float, np.ndarray]:
    features = get_features(bundle)
    x = input_df[features].copy()

    if "pipeline" in bundle:
        pipeline = bundle["pipeline"]

        # Case A: pipeline itself supports predict_proba.
        if hasattr(pipeline, "predict_proba"):
            prob = np.asarray(pipeline.predict_proba(x))[0]
            cluster_id = int(np.argmax(prob))
            confidence = float(np.max(prob))
            return cluster_id, confidence, prob

        # Case B: pipeline supports transform, and final model is elsewhere.
        if hasattr(pipeline, "predict"):
            pred = pipeline.predict(x)
            cluster_id = int(pred[0])
            prob = np.zeros(max(cluster_id + 1, 1), dtype=float)
            prob[cluster_id] = 1.0
            return cluster_id, 1.0, prob

    # Common dict format.
    if "imputer" in bundle and bundle["imputer"] is not None:
        x_arr = bundle["imputer"].transform(x)
    else:
        x_arr = x.values

    if "scaler" in bundle and bundle["scaler"] is not None:
        x_arr = bundle["scaler"].transform(x_arr)

    gmm = None
    for key in ["gmm", "model", "cluster_model"]:
        if key in bundle and bundle[key] is not None:
            gmm = bundle[key]
            break

    if gmm is None:
        raise KeyError(
            "找不到 GMM 模型。請確認 models/soft_cluster_pipeline.pkl 內有 gmm/model/cluster_model，"
            "或是它本身是 sklearn pipeline。"
        )

    prob = np.asarray(gmm.predict_proba(x_arr))[0]
    cluster_id = int(np.argmax(prob))
    confidence = float(np.max(prob))
    return cluster_id, confidence, prob


def confidence_level(confidence: float) -> str:
    if confidence >= 0.70:
        return "高信心"
    if confidence >= 0.40:
        return "中等信心"
    return "低信心"


def load_label_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        return {}

    label_col = None
    for candidate in ["cluster_label", "label", "name"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        return {}

    return {
        int(row["cluster"]): str(row[label_col])
        for _, row in df.iterrows()
    }


def load_cluster_feature_summary(path: Path, cluster_id: int, top_n: int = 3) -> str:
    if not path.exists():
        return ""

    df = pd.read_csv(path, index_col=0)

    # index may be str or int
    if cluster_id in df.index:
        row = df.loc[cluster_id]
    elif str(cluster_id) in df.index:
        row = df.loc[str(cluster_id)]
    else:
        return ""

    high = row.sort_values(ascending=False).head(top_n)
    low = row.sort_values(ascending=True).head(top_n)

    high_text = "、".join([f"{k}={v:.2f}" for k, v in high.items()])
    low_text = "、".join([f"{k}={v:.2f}" for k, v in low.items()])

    return f"高特徵：{high_text}；低特徵：{low_text}"


def build_input_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "case_name": "自訂輸入",
        "budget": args.budget,
        "available_time": args.available_time,
        "weather_badness": args.weather_badness,
        "social_context": args.social_context,
        "fatigue": args.fatigue,
        "spontaneity": args.spontaneity,
        "selected_indoor_score": args.indoor,
        "selected_cost_level": args.cost,
        "selected_photo_value": args.photo,
        "selected_nature_value": args.nature,
        "selected_culture_value": args.culture,
        "selected_food_value": args.food,
        "selected_popularity": args.popularity,
    }


def run_cases(cases: list[dict[str, float]], show_prob: bool) -> None:
    model_path = Path("models/soft_cluster_pipeline.pkl")
    label_path = Path("data/soft_cluster_label_summary.csv")
    feature_mean_path = Path("data/soft_cluster_feature_mean.csv")

    if not model_path.exists():
        raise FileNotFoundError(
            "找不到 models/soft_cluster_pipeline.pkl。請先執行：python cluster_interpretation_soft.py"
        )

    bundle = load_model_bundle(model_path)
    features = get_features(bundle)
    label_map = load_label_map(label_path)

    outputs = []

    for case in cases:
        input_df = pd.DataFrame([{f: case[f] for f in features}])
        cluster_id, confidence, prob = predict_cluster(bundle, input_df)

        label = label_map.get(cluster_id, "")

        outputs.append({
            "case_name": case.get("case_name", ""),
            "cluster": cluster_id,
            "cluster_label": label,
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level(confidence),
            "top_prob_clusters": ", ".join(
                [
                    f"C{idx}:{p:.3f}"
                    for idx, p in sorted(enumerate(prob), key=lambda x: x[1], reverse=True)[:3]
                ]
            ),
            "cluster_feature_summary": load_cluster_feature_summary(feature_mean_path, cluster_id),
        })

        if show_prob:
            print("\n" + "=" * 80)
            print(f"Case: {case.get('case_name', '')}")
            print("- input:")
            for f in features:
                print(f"  {DISPLAY_NAMES.get(f, f)} = {case[f]}")
            print("- probability:")
            for idx, p in sorted(enumerate(prob), key=lambda x: x[1], reverse=True):
                print(f"  Cluster {idx}: {p:.4f}")

    result_df = pd.DataFrame(outputs)

    print("\n" + "=" * 80)
    print("Cluster 測試結果")
    print("=" * 80)
    print(result_df.to_string(index=False))

    output_path = Path("data/cluster_input_test_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", action="store_true", help="使用內建測試案例")
    parser.add_argument("--show-prob", action="store_true", help="顯示每個 cluster 的完整 probability")

    parser.add_argument("--budget", type=float, default=3)
    parser.add_argument("--available-time", type=float, default=3)
    parser.add_argument("--weather-badness", type=float, default=0.5)
    parser.add_argument("--social-context", type=float, default=0.5)
    parser.add_argument("--fatigue", type=float, default=2.5)
    parser.add_argument("--spontaneity", type=float, default=2.5)

    parser.add_argument("--indoor", type=float, default=2.5)
    parser.add_argument("--cost", type=float, default=2.5)
    parser.add_argument("--photo", type=float, default=2.5)
    parser.add_argument("--nature", type=float, default=2.5)
    parser.add_argument("--culture", type=float, default=2.5)
    parser.add_argument("--food", type=float, default=2.5)
    parser.add_argument("--popularity", type=float, default=2.5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset:
        cases = PRESET_CASES
    else:
        cases = [build_input_from_args(args)]

    run_cases(cases=cases, show_prob=args.show_prob)


if __name__ == "__main__":
    main()
