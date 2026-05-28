"""
test_cluster_inputs.py

用途：
測試不同輸入參數下會對應到哪個 cluster。

這版支援兩種模式：
1. gmm：優先讀取 models/soft_cluster_pipeline.pkl，模擬 App 的 GMM 預測。
2. mean：不讀 pkl，改用 data/travel_behavior_soft_clustered.csv 或
   data/soft_cluster_feature_mean.csv 進行 cluster mean matching。
   若 pkl 因 pickle/joblib 相容問題讀取失敗，會自動 fallback 到 mean。

建議先用：
python test_cluster_inputs.py --preset --method mean --show-prob

若要嘗試和 App 一樣讀模型：
python test_cluster_inputs.py --preset --method gmm --show-prob
"""

from __future__ import annotations

import argparse
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


def softmax_from_distance(distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert distances to pseudo probabilities. Smaller distance = higher probability."""
    scores = -distances / max(temperature, 1e-9)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


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


def load_pickle_or_joblib(path: Path) -> Any:
    """
    Load model bundle.

    先嘗試 pickle；失敗後嘗試 joblib。
    某些環境下 pickle 會因自訂物件或版本差異出現 ModuleNotFoundError。
    """
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as pickle_error:
        try:
            import joblib
            return joblib.load(path)
        except Exception as joblib_error:
            raise RuntimeError(
                "無法讀取 models/soft_cluster_pipeline.pkl。\n"
                f"pickle error: {repr(pickle_error)}\n"
                f"joblib error: {repr(joblib_error)}\n"
                "建議改用 --method mean，或重新執行 python cluster_interpretation_soft.py 產生新的模型檔。"
            )


def get_features_from_bundle(bundle: Any) -> list[str]:
    if isinstance(bundle, dict):
        for key in ["features", "feature_names", "input_features"]:
            if key in bundle and bundle[key] is not None:
                return list(bundle[key])
    return FEATURES


def predict_with_gmm(case: dict[str, float]) -> tuple[int, float, np.ndarray]:
    model_path = Path("models/soft_cluster_pipeline.pkl")
    if not model_path.exists():
        raise FileNotFoundError("找不到 models/soft_cluster_pipeline.pkl")

    bundle = load_pickle_or_joblib(model_path)
    features = get_features_from_bundle(bundle)
    x = pd.DataFrame([{f: case[f] for f in features}])

    if not isinstance(bundle, dict):
        if hasattr(bundle, "predict_proba"):
            prob = np.asarray(bundle.predict_proba(x))[0]
            cluster_id = int(np.argmax(prob))
            return cluster_id, float(np.max(prob)), prob

        if hasattr(bundle, "predict"):
            pred = bundle.predict(x)
            cluster_id = int(pred[0])
            prob = np.zeros(cluster_id + 1)
            prob[cluster_id] = 1.0
            return cluster_id, 1.0, prob

        raise TypeError("模型檔不是 dict，也沒有 predict / predict_proba。")

    if "pipeline" in bundle:
        pipeline = bundle["pipeline"]
        if hasattr(pipeline, "predict_proba"):
            prob = np.asarray(pipeline.predict_proba(x))[0]
            cluster_id = int(np.argmax(prob))
            return cluster_id, float(np.max(prob)), prob

    x_arr = x.values

    if "imputer" in bundle and bundle["imputer"] is not None:
        x_arr = bundle["imputer"].transform(x_arr)

    if "scaler" in bundle and bundle["scaler"] is not None:
        x_arr = bundle["scaler"].transform(x_arr)

    gmm = None
    for key in ["gmm", "model", "cluster_model"]:
        if key in bundle and bundle[key] is not None:
            gmm = bundle[key]
            break

    if gmm is None:
        raise KeyError("模型檔中找不到 gmm/model/cluster_model。")

    prob = np.asarray(gmm.predict_proba(x_arr))[0]
    cluster_id = int(np.argmax(prob))
    return cluster_id, float(np.max(prob)), prob


def predict_with_cluster_mean(case: dict[str, float]) -> tuple[int, float, np.ndarray]:
    """
    不讀取 pkl，直接用訓練資料或 cluster heatmap 的平均值做近似判斷。

    優先使用 data/travel_behavior_soft_clustered.csv：
    - 由原始資料計算 mean/std
    - 將輸入轉成 z-score
    - 和每個 cluster 的 z-score mean 比距離

    若沒有該檔，則使用 data/soft_cluster_feature_mean.csv：
    - 假設該檔已是 z-score mean
    - 但輸入無法正確標準化，所以會退而求其次用簡化尺度
    """
    clustered_path = Path("data/travel_behavior_soft_clustered.csv")
    mean_path = Path("data/soft_cluster_feature_mean.csv")

    if clustered_path.exists():
        df = pd.read_csv(clustered_path)

        if "soft_cluster" not in df.columns:
            raise KeyError("data/travel_behavior_soft_clustered.csv 缺少 soft_cluster 欄位。")

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise KeyError(f"data/travel_behavior_soft_clustered.csv 缺少欄位：{missing}")

        train_x = df[FEATURES].astype(float)
        mean = train_x.mean()
        std = train_x.std().replace(0, 1)

        input_z = ((pd.Series({f: case[f] for f in FEATURES}) - mean) / std).values.astype(float)

        cluster_z_mean = (
            ((df[FEATURES].astype(float) - mean) / std)
            .assign(soft_cluster=df["soft_cluster"])
            .groupby("soft_cluster")[FEATURES]
            .mean()
            .sort_index()
        )

        centers = cluster_z_mean.values.astype(float)

    elif mean_path.exists():
        cluster_z_mean = pd.read_csv(mean_path, index_col=0).sort_index()
        # fallback: rough scaling only
        rough_mean = pd.Series({
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
        })
        rough_std = pd.Series({
            "budget": 1.2,
            "available_time": 1.2,
            "weather_badness": 0.25,
            "social_context": 0.25,
            "fatigue": 1.2,
            "spontaneity": 1.2,
            "selected_indoor_score": 1.2,
            "selected_cost_level": 1.2,
            "selected_photo_value": 1.2,
            "selected_nature_value": 1.2,
            "selected_culture_value": 1.2,
            "selected_food_value": 1.2,
            "selected_popularity": 1.2,
        })
        input_z = ((pd.Series({f: case[f] for f in FEATURES}) - rough_mean) / rough_std).values.astype(float)
        centers = cluster_z_mean[FEATURES].values.astype(float)

    else:
        raise FileNotFoundError(
            "找不到 data/travel_behavior_soft_clustered.csv 或 data/soft_cluster_feature_mean.csv。"
            "請先執行 python cluster_interpretation_soft.py。"
        )

    distances = np.linalg.norm(centers - input_z.reshape(1, -1), axis=1)

    # temperature 用距離中位數，避免 probability 過度尖銳或過度平均
    temperature = max(float(np.median(distances)), 1e-6)
    prob = softmax_from_distance(distances, temperature=temperature)

    cluster_id = int(np.argmax(prob))
    confidence = float(np.max(prob))
    return cluster_id, confidence, prob


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


def run_cases(cases: list[dict[str, float]], method: str, show_prob: bool) -> None:
    label_map = load_label_map(Path("data/soft_cluster_label_summary.csv"))
    feature_mean_path = Path("data/soft_cluster_feature_mean.csv")

    rows = []

    for case in cases:
        used_method = method

        if method == "gmm":
            cluster_id, confidence, prob = predict_with_gmm(case)
        elif method == "mean":
            cluster_id, confidence, prob = predict_with_cluster_mean(case)
        elif method == "auto":
            try:
                cluster_id, confidence, prob = predict_with_gmm(case)
                used_method = "gmm"
            except Exception as error:
                print(f"[warning] GMM 讀取或預測失敗，改用 mean matching。原因：{error}")
                cluster_id, confidence, prob = predict_with_cluster_mean(case)
                used_method = "mean"
        else:
            raise ValueError("method 只能是 auto、gmm 或 mean。")

        label = label_map.get(cluster_id, "")

        rows.append({
            "case_name": case.get("case_name", ""),
            "method": used_method,
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
            print(f"Method: {used_method}")
            print("- input:")
            for f in FEATURES:
                print(f"  {DISPLAY_NAMES.get(f, f)} = {case[f]}")
            print("- probability:")
            for idx, p in sorted(enumerate(prob), key=lambda x: x[1], reverse=True):
                print(f"  Cluster {idx}: {p:.4f}")

    result_df = pd.DataFrame(rows)

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

    parser.add_argument(
        "--method",
        choices=["auto", "gmm", "mean"],
        default="auto",
        help="auto: 先用 GMM，失敗改用 mean；gmm: 強制讀模型；mean: 不讀模型，用 cluster 平均值測試。",
    )

    parser.add_argument("--preset", action="store_true", help="使用內建測試案例")
    parser.add_argument("--show-prob", action="store_true", help="顯示每個 cluster 的 probability")

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

    run_cases(cases=cases, method=args.method, show_prob=args.show_prob)


if __name__ == "__main__":
    main()
