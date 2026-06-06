"""
test_cluster_inputs.py

用途：
用 0~7 八組固定測試案例，快速確認目前 GMM soft clustering 對不同使用者輸入的分類結果。

本版重點：
1. 會讀取 models/soft_cluster_pipeline.pkl，不使用 mean matching fallback。
2. 若舊版 pkl 依賴 features.py，本程式會建立最小相容 features module。
3. 不使用 features.__getattr__，避免干擾 pickle 內部反序列化。
4. 優先使用 joblib.load；失敗後再使用 pickle.load。
5. 若 pkl 真的無法讀取，會顯示明確錯誤原因。

執行：
python test_cluster_inputs.py --preset --show-prob
python test_cluster_inputs.py --preset --case 3 --show-prob
"""

from __future__ import annotations

import argparse
import pickle
import pickletools
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALL_INPUT_FEATURES = [
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

PREFERENCE_FEATURES = [
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
        "case_id": 0,
        "case_name": "自然景點導向型",
        "expected_type": "自然景點導向型",
        "budget": 1000,
        "available_time": 5,
        "weather_badness": 0.0,
        "social_context": 0.5,
        "fatigue": 0.5,
        "spontaneity": 2,
        "selected_indoor_score": 0,
        "selected_cost_level": 0.5,
        "selected_photo_value": 2,
        "selected_nature_value": 5,
        "selected_culture_value": 0,
        "selected_food_value": 0,
        "selected_popularity": 1.5,
    },
    {
        "case_id": 1,
        "case_name": "室內文化型",
        "expected_type": "室內文化型",
        "budget": 1200,
        "available_time": 3,
        "weather_badness": 0.9,
        "social_context": 0.2,
        "fatigue": 4,
        "spontaneity": 1,
        "selected_indoor_score": 5,
        "selected_cost_level": 1,
        "selected_photo_value": 1,
        "selected_nature_value": 0,
        "selected_culture_value": 5,
        "selected_food_value": 1,
        "selected_popularity": 2,
    },
    {
        "case_id": 2,
        "case_name": "美食導向型",
        "expected_type": "美食導向型",
        "budget": 1500,
        "available_time": 2.5,
        "weather_badness": 0.5,
        "social_context": 1,
        "fatigue": 2,
        "spontaneity": 3,
        "selected_indoor_score": 3,
        "selected_cost_level": 3,
        "selected_photo_value": 1,
        "selected_nature_value": 0,
        "selected_culture_value": 0,
        "selected_food_value": 5,
        "selected_popularity": 2,
    },
    {
        "case_id": 3,
        "case_name": "拍照打卡型",
        "expected_type": "拍照打卡型",
        "budget": 1800,
        "available_time": 4,
        "weather_badness": 0.2,
        "social_context": 1,
        "fatigue": 1,
        "spontaneity": 3,
        "selected_indoor_score": 2,
        "selected_cost_level": 2,
        "selected_photo_value": 5,
        "selected_nature_value": 1,
        "selected_culture_value": 1,
        "selected_food_value": 1,
        "selected_popularity": 5,
    },
    {
        "case_id": 4,
        "case_name": "高消費型",
        "expected_type": "高消費型",
        "budget": 5000,
        "available_time": 4,
        "weather_badness": 0.3,
        "social_context": 1,
        "fatigue": 1,
        "spontaneity": 3,
        "selected_indoor_score": 3,
        "selected_cost_level": 5,
        "selected_photo_value": 3,
        "selected_nature_value": 1,
        "selected_culture_value": 2,
        "selected_food_value": 3,
        "selected_popularity": 4,
    },
    {
        "case_id": 5,
        "case_name": "熱門景點型",
        "expected_type": "熱門景點型",
        "budget": 2000,
        "available_time": 4,
        "weather_badness": 0.2,
        "social_context": 1,
        "fatigue": 1,
        "spontaneity": 4,
        "selected_indoor_score": 2,
        "selected_cost_level": 2,
        "selected_photo_value": 3,
        "selected_nature_value": 1,
        "selected_culture_value": 1,
        "selected_food_value": 1,
        "selected_popularity": 5,
    },
    {
        "case_id": 6,
        "case_name": "低消費輕旅行型",
        "expected_type": "低消費輕旅行型",
        "budget": 500,
        "available_time": 2,
        "weather_badness": 0.2,
        "social_context": 0,
        "fatigue": 2,
        "spontaneity": 4,
        "selected_indoor_score": 1,
        "selected_cost_level": 0,
        "selected_photo_value": 1,
        "selected_nature_value": 2,
        "selected_culture_value": 1,
        "selected_food_value": 1,
        "selected_popularity": 1,
    },
    {
        "case_id": 7,
        "case_name": "中性模糊型",
        "expected_type": "中性模糊型",
        "budget": 1500,
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


# ============================================================
# Safe compatibility for old features.py references
# ============================================================

class CompatTransformer:
    """
    Minimal sklearn-compatible transformer used only when old pickle references
    custom classes from a removed features.py file.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setstate__(self, state):
        # Accept any pickle state shape.
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["pickle_state"] = state

    def __getstate__(self):
        return self.__dict__

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Try to behave as a feature selector if old state contains feature names.
        possible_attrs = [
            "features",
            "feature_names",
            "input_features",
            "columns",
            "cols",
            "selected_features",
            "feature_cols",
            "model_features",
        ]

        cols = None
        for attr in possible_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    cols = list(val)
                    break
                except TypeError:
                    pass

        if cols is not None and hasattr(X, "loc"):
            existing = [c for c in cols if c in X.columns]
            if existing:
                return X.loc[:, existing]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def scan_features_references(path: Path) -> set[str]:
    """
    Scan pickle opcodes and find symbols referenced from module 'features'.

    This works for many pickle files. If scanning fails, we still define common names.
    """
    refs: set[str] = set()

    try:
        data = path.read_bytes()
        last_string = None

        for op, arg, _pos in pickletools.genops(data):
            if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING"}:
                s = str(arg)
                if last_string == "features":
                    refs.add(s)
                last_string = s
            elif op.name == "GLOBAL":
                # arg format usually: "module name"
                parts = str(arg).split()
                if len(parts) == 2 and parts[0] == "features":
                    refs.add(parts[1])
    except Exception:
        pass

    return refs


def install_features_module_for_pickle(path: Path) -> None:
    """
    Create a minimal in-memory module named 'features' only when needed.
    No __getattr__ is used, because it can interfere with pickle internals.
    """
    if "features" in sys.modules:
        return

    module = types.ModuleType("features")
    module.__file__ = "<compat features module>"
    module.__package__ = ""

    # constants that old code may have stored/referenced
    module.ALL_INPUT_FEATURES = ALL_INPUT_FEATURES
    module.FEATURES = ALL_INPUT_FEATURES
    module.MODEL_FEATURES = PREFERENCE_FEATURES
    module.PREFERENCE_FEATURES = PREFERENCE_FEATURES
    module.BEHAVIOR_FEATURES = PREFERENCE_FEATURES

    common_class_names = {
        "FeatureSelector",
        "ColumnSelector",
        "DataFrameSelector",
        "FeatureEngineer",
        "FeatureTransformer",
        "BehaviorFeatureSelector",
        "PreferenceFeatureSelector",
        "TravelFeatureSelector",
    }

    referenced = scan_features_references(path)
    class_names = common_class_names | {name for name in referenced if name and name[:1].isupper()}

    for name in class_names:
        cls = type(name, (CompatTransformer,), {"__module__": "features"})
        setattr(module, name, cls)

    sys.modules["features"] = module


def load_model_object(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            "找不到 models/soft_cluster_pipeline.pkl。請先執行：python cluster_interpretation_soft.py"
        )

    install_features_module_for_pickle(path)

    errors = []

    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        errors.append(f"joblib.load failed: {type(exc).__name__}: {exc}")

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        errors.append(f"pickle.load failed: {type(exc).__name__}: {exc}")

    error_text = "\n".join(errors)
    raise RuntimeError(
        "無法讀取 models/soft_cluster_pipeline.pkl。\n"
        "這通常代表模型檔是由不同版本的程式或套件儲存，且裡面引用了目前不存在或不相容的自訂物件。\n\n"
        f"{error_text}\n\n"
        "請確認：\n"
        "1. 目前資料夾是否有產生該 pkl 的原始 features.py 或相關自訂模組。\n"
        "2. 或重新執行 python cluster_interpretation_soft.py 產生新版 pkl。\n"
    )


def load_model_bundle(path: Path) -> dict[str, Any]:
    obj = load_model_object(path)

    if isinstance(obj, dict):
        return obj

    return {"pipeline": obj}


def get_features(bundle: dict[str, Any]) -> list[str]:
    for key in ["features", "feature_names", "input_features"]:
        if key in bundle and bundle[key] is not None:
            return list(bundle[key])

    pipeline = bundle.get("pipeline")
    if pipeline is not None:
        for attr in ["feature_names_in_", "features", "feature_names", "input_features"]:
            if hasattr(pipeline, attr):
                val = getattr(pipeline, attr)
                if val is not None:
                    return list(val)

    return PREFERENCE_FEATURES


# ============================================================
# Prediction
# ============================================================

def predict_cluster(bundle: dict[str, Any], input_df: pd.DataFrame) -> tuple[int, float, np.ndarray, list[str]]:
    features = get_features(bundle)

    missing_features = [f for f in features if f not in input_df.columns]
    if missing_features:
        raise KeyError(f"輸入資料缺少模型需要的欄位：{missing_features}")

    x = input_df[features].copy()

    if "pipeline" in bundle:
        pipeline = bundle["pipeline"]

        if hasattr(pipeline, "predict_proba"):
            prob = np.asarray(pipeline.predict_proba(x))[0]
            cluster_id = int(np.argmax(prob))
            confidence = float(np.max(prob))
            return cluster_id, confidence, prob, features

        if hasattr(pipeline, "predict"):
            pred = pipeline.predict(x)
            cluster_id = int(pred[0])
            prob = np.zeros(max(cluster_id + 1, 1), dtype=float)
            prob[cluster_id] = 1.0
            return cluster_id, 1.0, prob, features

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
            "找不到 GMM 模型。請確認 pkl 內有 gmm/model/cluster_model，或 pkl 本身是 sklearn pipeline。"
        )

    prob = np.asarray(gmm.predict_proba(x_arr))[0]
    cluster_id = int(np.argmax(prob))
    confidence = float(np.max(prob))
    return cluster_id, confidence, prob, features


# ============================================================
# Labels and summaries
# ============================================================

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

    cluster_col = None
    for candidate in ["cluster", "soft_cluster"]:
        if candidate in df.columns:
            cluster_col = candidate
            break

    if cluster_col is None:
        return {}

    label_col = None
    for candidate in ["cluster_label", "label", "name"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        return {}

    return {int(row[cluster_col]): str(row[label_col]) for _, row in df.iterrows()}


def load_cluster_feature_summary(path: Path, cluster_id: int, top_n: int = 3) -> str:
    if not path.exists():
        return ""

    df = pd.read_csv(path)

    for candidate in ["soft_cluster", "cluster"]:
        if candidate in df.columns:
            df = df.set_index(candidate)
            break

    if "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")

    if cluster_id in df.index:
        row = df.loc[cluster_id]
    elif str(cluster_id) in df.index.astype(str):
        row = df.loc[df.index.astype(str) == str(cluster_id)].iloc[0]
    else:
        return ""

    numeric_row = pd.to_numeric(row, errors="coerce").dropna()
    if numeric_row.empty:
        return ""

    high = numeric_row.sort_values(ascending=False).head(top_n)
    low = numeric_row.sort_values(ascending=True).head(top_n)

    high_text = "、".join([f"{k}={v:.2f}" for k, v in high.items()])
    low_text = "、".join([f"{k}={v:.2f}" for k, v in low.items()])

    return f"高特徵：{high_text}；低特徵：{low_text}"


def format_top_prob(prob: np.ndarray, top_n: int = 3) -> str:
    return ", ".join(
        [f"C{idx}:{p:.3f}" for idx, p in sorted(enumerate(prob), key=lambda x: x[1], reverse=True)[:top_n]]
    )


def possible_match(expected_type: str, actual_label: str, confidence: float) -> str:
    if not actual_label:
        return "未判定"

    if expected_type == "中性模糊型":
        return "合理" if confidence < 0.70 else "需檢查"

    keywords = {
        "自然景點導向型": ["自然", "戶外", "景觀", "公園"],
        "室內文化型": ["室內", "文化", "博物館", "展覽"],
        "美食導向型": ["美食", "餐飲", "食"],
        "拍照打卡型": ["拍照", "打卡", "熱門"],
        "高消費型": ["高消費", "消費", "熱門"],
        "熱門景點型": ["熱門", "打卡"],
        "低消費輕旅行型": ["低消費", "低預算", "自然", "輕旅行"],
    }

    return "合理" if any(k in actual_label for k in keywords.get(expected_type, [])) else "需檢查"


# ============================================================
# Inputs and CLI
# ============================================================

def build_input_from_args(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "case_id": "custom",
        "case_name": "自訂輸入",
        "expected_type": "自訂輸入",
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


def select_cases(args: argparse.Namespace) -> list[dict[str, float | str]]:
    if not args.preset:
        return [build_input_from_args(args)]

    if args.case is None:
        return PRESET_CASES

    selected = [c for c in PRESET_CASES if int(c["case_id"]) == args.case]
    if not selected:
        raise ValueError(f"找不到 case_id={args.case}。可用範圍為 0~7。")
    return selected


def run_cases(cases: list[dict[str, float | str]], show_prob: bool) -> None:
    model_path = Path("models/soft_cluster_pipeline.pkl")
    label_path = Path("data/soft_cluster_label_summary.csv")
    feature_mean_path = Path("data/soft_cluster_feature_mean.csv")

    bundle = load_model_bundle(model_path)
    features = get_features(bundle)
    label_map = load_label_map(label_path)

    print("=" * 80)
    print("Loaded model pkl successfully")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print("Model input features:")
    print(", ".join(features))
    print()

    outputs = []

    for case in cases:
        input_df = pd.DataFrame([{f: case[f] for f in ALL_INPUT_FEATURES}])
        cluster_id, confidence, prob, used_features = predict_cluster(bundle, input_df)

        label = label_map.get(cluster_id, "")
        expected_type = str(case.get("expected_type", ""))

        outputs.append({
            "case_id": case.get("case_id", ""),
            "case_name": case.get("case_name", ""),
            "expected_type": expected_type,
            "predicted_cluster": cluster_id,
            "cluster_label": label,
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level(confidence),
            "top_prob_clusters": format_top_prob(prob, top_n=3),
            "match_check": possible_match(expected_type, label, confidence),
            "cluster_feature_summary": load_cluster_feature_summary(feature_mean_path, cluster_id),
        })

        if show_prob:
            print("\n" + "=" * 80)
            print(f"Case {case.get('case_id', '')}: {case.get('case_name', '')}")
            print("- expected type:")
            print(f"  {expected_type}")
            print("- input:")
            for f in ALL_INPUT_FEATURES:
                print(f"  {DISPLAY_NAMES.get(f, f)} = {case[f]}")
            print("- model-used features:")
            for f in used_features:
                print(f"  {DISPLAY_NAMES.get(f, f)} = {case[f]}")
            print("- probability:")
            for idx, p in sorted(enumerate(prob), key=lambda x: x[1], reverse=True):
                print(f"  Cluster {idx}: {p:.4f}")
            print("- predicted:")
            print(f"  cluster={cluster_id}, label={label}, confidence={confidence:.4f}")

    result_df = pd.DataFrame(outputs)

    print("\n" + "=" * 80)
    print("0~7 Cluster 測試結果")
    print("=" * 80)
    print(result_df.to_string(index=False))

    output_path = Path("data/cluster_input_test_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path}")

    print("\n判讀提醒：")
    print("1. cluster ID 不一定要等於 case_id，因為 GMM 重訓後 cluster 編號可能改變。")
    print("2. 請以 cluster_label、top_prob_clusters、cluster_feature_summary 和 confidence 綜合判讀。")
    print("3. match_check 只是粗略輔助，不是正式準確率。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", action="store_true", help="使用 0~7 內建測試案例")
    parser.add_argument("--case", type=int, default=None, help="只測試指定 case_id，範圍 0~7")
    parser.add_argument("--show-prob", action="store_true", help="顯示每個 cluster 的完整 probability")

    parser.add_argument("--budget", type=float, default=1500)
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
    cases = select_cases(args)
    run_cases(cases=cases, show_prob=args.show_prob)


if __name__ == "__main__":
    main()
