"""
visualize_pipeline_data.py

用途：
視覺化檢查 travel recommendation project pipeline 產生的所有主要資料。

放置位置：
請把本檔案放在專案根目錄，也就是和 app.py、generate_data.py、
cluster_interpretation_soft.py 同一層。

執行：
python visualize_pipeline_data.py

輸出：
pipeline_audit/
├── pipeline_data_audit.html
└── figures/*.png

注意：
這支程式只讀取資料與產生圖表，不會修改 data/、models/ 或 figures/ 內原始檔案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Basic paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "pipeline_audit"
OUTPUT_FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Expected pipeline files
# =========================

EXPECTED_FILES = {
    "01_user_behavior": DATA_DIR / "travel_behavior.csv",
    "02_clustered_users": DATA_DIR / "travel_behavior_soft_clustered.csv",
    "03_cluster_feature_mean": DATA_DIR / "soft_cluster_feature_mean.csv",
    "04_cluster_interpretation": DATA_DIR / "soft_cluster_interpretation.csv",
    "05_cluster_label_summary": DATA_DIR / "soft_cluster_label_summary.csv",
    "06_attractions": DATA_DIR / "attractions.csv",
    "07_user_interactions": DATA_DIR / "user_interactions.csv",
    "08_all_recommendation_candidates": DATA_DIR / "all_recommendation_candidates.csv",
    "09_recommendations": DATA_DIR / "recommendations.csv",
    "10_method_comparison": DATA_DIR / "recommendation_method_comparison.csv",
    "11_model_pipeline": MODELS_DIR / "soft_cluster_pipeline.pkl",
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

ATTRACTION_FEATURES = [
    "indoor_score",
    "cost_level",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity",
    "estimated_time",
]

SCORE_COLUMNS = [
    "personal_score",
    "cluster_score",
    "collaborative_score",
    "hybrid_score",
    "recommendation_score",
]


# =========================
# Utility functions
# =========================

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def existing_columns(df: Optional[pd.DataFrame], cols: List[str]) -> List[str]:
    if df is None:
        return []
    return [c for c in cols if c in df.columns]


def save_current_fig(name: str) -> str:
    path = OUTPUT_FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return f"figures/{name}"


def dataframe_preview_html(df: Optional[pd.DataFrame], max_rows: int = 8) -> str:
    if df is None:
        return "<p class='missing'>File not found or failed to read.</p>"
    if df.empty:
        return "<p class='warning'>DataFrame is empty.</p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def dataframe_profile(df: Optional[pd.DataFrame], name: str) -> str:
    if df is None:
        return f"""
        <section>
        <h3>{html.escape(name)}</h3>
        <p class="missing">Missing or unreadable.</p>
        </section>
        """

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    dup_count = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_summary = ""
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().T.round(4).to_html()

    missing_html = (
        missing.to_frame("missing_count").to_html()
        if len(missing) > 0
        else "<p>No missing values detected.</p>"
    )

    return f"""
    <section>
      <h3>{html.escape(name)}</h3>
      <table>
        <tr><th>Rows</th><td>{len(df):,}</td></tr>
        <tr><th>Columns</th><td>{len(df.columns):,}</td></tr>
        <tr><th>Duplicated rows</th><td>{dup_count:,}</td></tr>
        <tr><th>Numeric columns</th><td>{len(numeric_cols)}</td></tr>
        <tr><th>Non-numeric columns</th><td>{len(object_cols)}</td></tr>
      </table>
      <details>
        <summary>Columns</summary>
        <p class="mono">{html.escape(", ".join(df.columns))}</p>
      </details>
      <details>
        <summary>Missing values</summary>
        {missing_html}
      </details>
      <details>
        <summary>Numeric summary</summary>
        {numeric_summary if numeric_summary else "<p>No numeric columns.</p>"}
      </details>
      <details>
        <summary>Preview</summary>
        {dataframe_preview_html(df)}
      </details>
    </section>
    """


# =========================
# Load data
# =========================

dataframes: Dict[str, Optional[pd.DataFrame]] = {
    key: safe_read_csv(path)
    for key, path in EXPECTED_FILES.items()
    if path.suffix.lower() == ".csv"
}

generated_figures: List[Tuple[str, str]] = []


# =========================
# Figure generation
# =========================

def plot_file_status() -> None:
    labels = list(EXPECTED_FILES.keys())
    status = [1 if path.exists() else 0 for path in EXPECTED_FILES.values()]

    plt.figure(figsize=(10, 4.8))
    y = np.arange(len(labels))
    plt.barh(y, status)
    plt.yticks(y, labels)
    plt.xticks([0, 1], ["missing", "exists"])
    plt.xlabel("File status")
    plt.title("Pipeline expected file status")

    for i, v in enumerate(status):
        plt.text((v + 0.02) if v else 0.02, i, "exists" if v else "missing", va="center")

    generated_figures.append(("Pipeline file status", save_current_fig("01_pipeline_file_status.png")))


def plot_cluster_counts() -> None:
    df = dataframes.get("02_clustered_users")
    if df is None or "soft_cluster" not in df.columns:
        return

    counts = df["soft_cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 4.5))
    counts.plot(kind="bar")
    plt.xlabel("soft_cluster")
    plt.ylabel("count")
    plt.title("User count by soft cluster")
    generated_figures.append(("User count by soft cluster", save_current_fig("02_cluster_counts.png")))


def plot_cluster_confidence() -> None:
    df = dataframes.get("02_clustered_users")
    if df is None or "soft_cluster_confidence" not in df.columns:
        return

    plt.figure(figsize=(8, 4.5))
    df["soft_cluster_confidence"].dropna().hist(bins=30)
    plt.xlabel("soft_cluster_confidence")
    plt.ylabel("count")
    plt.title("Cluster confidence distribution")
    generated_figures.append(("Cluster confidence distribution", save_current_fig("03_cluster_confidence_distribution.png")))


def plot_cluster_probability_mean() -> None:
    df = dataframes.get("02_clustered_users")
    if df is None:
        return

    prob_cols = [c for c in df.columns if c.startswith("cluster_prob_")]
    if not prob_cols:
        return

    means = df[prob_cols].mean().sort_index()
    plt.figure(figsize=(9, 4.5))
    means.plot(kind="bar")
    plt.xlabel("cluster probability column")
    plt.ylabel("mean probability")
    plt.title("Mean cluster probability over users")
    generated_figures.append(("Mean cluster probability", save_current_fig("04_cluster_probability_mean.png")))


def plot_heatmap_from_feature_mean() -> None:
    df = dataframes.get("03_cluster_feature_mean")
    if df is None or df.empty:
        return

    plot_df = df.copy()
    for c in ["soft_cluster", "cluster", "label"]:
        if c in plot_df.columns:
            plot_df = plot_df.set_index(c)
            break

    numeric_df = plot_df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    plt.figure(figsize=(10, max(4, 0.45 * len(numeric_df))))
    plt.imshow(numeric_df.values, aspect="auto")
    plt.colorbar(label="z_mean / standardized mean")
    plt.xticks(np.arange(len(numeric_df.columns)), numeric_df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(numeric_df.index)), numeric_df.index.astype(str))
    plt.title("Cluster feature mean heatmap")

    for i in range(numeric_df.shape[0]):
        for j in range(numeric_df.shape[1]):
            plt.text(j, i, f"{numeric_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    generated_figures.append(("Cluster feature mean heatmap", save_current_fig("05_heatmap_from_feature_mean.png")))


def plot_method_comparison() -> None:
    df = dataframes.get("10_method_comparison")
    if df is None or df.empty:
        return

    if "method" in df.columns:
        numeric_cols = [c for c in ["mean_similarity", "random_baseline", "recommendation_lift"] if c in df.columns]
        for col in numeric_cols:
            plt.figure(figsize=(8, 4.5))
            plt.bar(df["method"].astype(str), df[col])
            plt.xlabel("method")
            plt.ylabel(col)
            plt.title(f"Recommendation method comparison: {col}")
            plt.xticks(rotation=20, ha="right")
            generated_figures.append((f"Method comparison: {col}", save_current_fig(f"06_recommendation_method_comparison_{col}.png")))
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        plt.figure(figsize=(9, 4.5))
        numeric_df.plot(kind="bar")
        plt.title("Recommendation method comparison")
        plt.ylabel("value")
        generated_figures.append(("Recommendation method comparison", save_current_fig("06_recommendation_method_comparison.png")))


def plot_attraction_type_counts() -> None:
    df = dataframes.get("06_attractions")
    if df is None:
        return

    type_col = None
    for c in ["attraction_type", "type", "category", "osm_category"]:
        if c in df.columns:
            type_col = c
            break

    if type_col is None:
        return

    counts = df[type_col].astype(str).value_counts().head(20)
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.xlabel(type_col)
    plt.ylabel("count")
    plt.title("Top attraction types")
    plt.xticks(rotation=45, ha="right")
    generated_figures.append(("Top attraction types", save_current_fig("07_attraction_type_counts.png")))


def plot_attraction_feature_distribution() -> None:
    df = dataframes.get("06_attractions")
    cols = existing_columns(df, ATTRACTION_FEATURES)
    if df is None or not cols:
        return

    plt.figure(figsize=(10, 5))
    values_by_col = [df[c].dropna().values for c in cols]
    plt.boxplot(values_by_col, labels=cols, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("value")
    plt.title("Attraction feature distribution")
    generated_figures.append(("Attraction feature distribution", save_current_fig("08_attraction_feature_distribution.png")))


def plot_user_preference_distribution() -> None:
    df = dataframes.get("01_user_behavior")
    cols = existing_columns(df, PREFERENCE_FEATURES)
    if df is None or not cols:
        return

    plt.figure(figsize=(10, 5))
    values_by_col = [df[c].dropna().values for c in cols]
    plt.boxplot(values_by_col, labels=cols, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("value")
    plt.title("User preference feature distribution")
    generated_figures.append(("User preference feature distribution", save_current_fig("09_user_preference_distribution.png")))


def plot_preference_correlation() -> None:
    df = dataframes.get("01_user_behavior")
    cols = existing_columns(df, PREFERENCE_FEATURES + CONTEXT_FEATURES)
    if df is None or len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="correlation")
    plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(corr.index)), corr.index)
    plt.title("User feature correlation matrix")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)

    generated_figures.append(("User feature correlation matrix", save_current_fig("10_preference_correlation.png")))


def plot_interaction_rating_distribution() -> None:
    df = dataframes.get("07_user_interactions")
    if df is None:
        return

    rating_col = None
    for c in ["rating", "score", "interaction_score"]:
        if c in df.columns:
            rating_col = c
            break

    if rating_col:
        plt.figure(figsize=(8, 4.5))
        df[rating_col].dropna().hist(bins=30)
        plt.xlabel(rating_col)
        plt.ylabel("count")
        plt.title("Interaction rating distribution")
        generated_figures.append(("Interaction rating distribution", save_current_fig("11_interaction_rating_distribution.png")))


def plot_recommendation_score_distribution() -> None:
    df = dataframes.get("08_all_recommendation_candidates")
    cols = existing_columns(df, SCORE_COLUMNS)
    if df is None or not cols:
        df = dataframes.get("09_recommendations")
        cols = existing_columns(df, SCORE_COLUMNS)

    if df is None or not cols:
        return

    plt.figure(figsize=(10, 5))
    values_by_col = [df[c].dropna().values for c in cols]
    plt.boxplot(values_by_col, labels=cols, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("score")
    plt.title("Recommendation score distribution")
    generated_figures.append(("Recommendation score distribution", save_current_fig("12_recommendation_score_distribution.png")))


def plot_user_cluster_pca() -> None:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return

    df = dataframes.get("02_clustered_users")
    if df is None or "soft_cluster" not in df.columns:
        return

    cols = existing_columns(df, PREFERENCE_FEATURES)
    if len(cols) < 2:
        return

    X = df[cols].dropna()
    if len(X) < 5:
        return

    clusters = df.loc[X.index, "soft_cluster"].astype(str)

    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    unique_clusters = sorted(clusters.unique(), key=lambda x: str(x))
    for cl in unique_clusters:
        mask = clusters == cl
        plt.scatter(pcs[mask, 0], pcs[mask, 1], s=10, alpha=0.6, label=str(cl))
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("User preference PCA by soft cluster")
    plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    generated_figures.append(("User preference PCA by soft cluster", save_current_fig("13_user_cluster_pca.png")))


plot_file_status()
plot_cluster_counts()
plot_cluster_confidence()
plot_cluster_probability_mean()
plot_heatmap_from_feature_mean()
plot_method_comparison()
plot_attraction_type_counts()
plot_attraction_feature_distribution()
plot_user_preference_distribution()
plot_preference_correlation()
plot_interaction_rating_distribution()
plot_recommendation_score_distribution()
plot_user_cluster_pca()


# =========================
# Consistency checks
# =========================

checks: List[Tuple[str, str, str]] = []

def add_check(name: str, status: str, detail: str) -> None:
    checks.append((name, status, detail))

for key, path in EXPECTED_FILES.items():
    add_check(
        f"File exists: {key}",
        "PASS" if path.exists() else "WARN",
        str(path.relative_to(PROJECT_ROOT)) if path.exists() else f"Missing: {path.relative_to(PROJECT_ROOT)}",
    )

clustered_df = dataframes.get("02_clustered_users")
if clustered_df is not None:
    pref_cols = existing_columns(clustered_df, PREFERENCE_FEATURES)
    context_cols = existing_columns(clustered_df, CONTEXT_FEATURES)
    add_check(
        "User preference columns in clustered data",
        "PASS" if len(pref_cols) == len(PREFERENCE_FEATURES) else "WARN",
        f"Found {len(pref_cols)}/{len(PREFERENCE_FEATURES)} preference columns.",
    )
    add_check(
        "Context columns retained for display",
        "PASS" if len(context_cols) > 0 else "INFO",
        f"Found context columns: {', '.join(context_cols) if context_cols else 'none'}",
    )

if clustered_df is not None and "soft_cluster_confidence" in clustered_df.columns:
    conf = clustered_df["soft_cluster_confidence"].dropna()
    if len(conf):
        add_check(
            "Cluster confidence range",
            "PASS" if conf.between(0, 1).all() else "WARN",
            f"min={conf.min():.4f}, mean={conf.mean():.4f}, max={conf.max():.4f}",
        )

att_df = dataframes.get("06_attractions")
if att_df is not None:
    att_cols = existing_columns(att_df, ATTRACTION_FEATURES)
    add_check(
        "Attraction feature columns",
        "PASS" if len(att_cols) >= 6 else "WARN",
        f"Found columns: {', '.join(att_cols)}",
    )

cmp_df = dataframes.get("10_method_comparison")
if cmp_df is not None and "method" in cmp_df.columns:
    methods = set(cmp_df["method"].astype(str))
    expected_methods = {"personal_only", "cluster_only", "collaborative_only", "hybrid"}
    add_check(
        "Recommendation methods present",
        "PASS" if expected_methods.issubset(methods) else "WARN",
        f"Found methods: {', '.join(sorted(methods))}",
    )


# =========================
# HTML report
# =========================

fig_html = "\n".join(
    f"""
    <div class="figure-card">
      <h3>{html.escape(title)}</h3>
      <img src="{html.escape(rel_path)}" alt="{html.escape(title)}">
    </div>
    """
    for title, rel_path in generated_figures
)

profiles_html = "\n".join(
    dataframe_profile(df, key)
    for key, df in dataframes.items()
)

checks_df = pd.DataFrame(checks, columns=["check", "status", "detail"])
checks_html = checks_df.to_html(index=False, escape=True)

file_status_rows = []
for key, path in EXPECTED_FILES.items():
    exists = path.exists()
    file_status_rows.append({
        "key": key,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": exists,
        "size_kb": round(path.stat().st_size / 1024, 2) if exists and path.is_file() else None,
    })
file_status_df = pd.DataFrame(file_status_rows)

html_report = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>Pipeline Data Audit</title>
<style>
body {{
  font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif;
  line-height: 1.65;
  margin: 0;
  background: #f5f5f5;
  color: #222;
}}
.container {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 20px 60px;
}}
h1, h2, h3 {{
  margin-top: 0;
}}
.card, section {{
  background: white;
  padding: 20px 24px;
  margin-bottom: 18px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 14px;
  margin: 10px 0 18px;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 7px 9px;
  vertical-align: top;
}}
th {{
  background: #f0f0f0;
}}
img {{
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: white;
}}
.figure-card {{
  background: white;
  padding: 18px 20px;
  margin-bottom: 18px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}
.mono {{
  font-family: Consolas, monospace;
  word-break: break-all;
}}
.missing {{
  color: #a60000;
  font-weight: 700;
}}
.warning {{
  color: #945500;
  font-weight: 700;
}}
summary {{
  cursor: pointer;
  font-weight: 700;
  margin: 8px 0;
}}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Pipeline Data Audit</h1>
    <p>本報告用來視覺化確認 travel recommendation project 的資料 pipeline 是否完整。</p>
    <p class="mono">Project root: {html.escape(str(PROJECT_ROOT))}</p>
  </div>

  <div class="card">
    <h2>1. Expected file status</h2>
    {file_status_df.to_html(index=False, escape=True)}
  </div>

  <div class="card">
    <h2>2. Consistency checks</h2>
    {checks_html}
  </div>

  <div class="card">
    <h2>3. Generated visualizations</h2>
    <p>以下圖表用於檢查 cluster、推薦分數、OSM 景點特徵、互動資料與方法比較結果。</p>
  </div>

  {fig_html}

  <div class="card">
    <h2>4. DataFrame profiles</h2>
    <p>以下區塊提供每個主要 CSV 的 rows、columns、missing values、numeric summary 與前幾筆資料預覽。</p>
  </div>

  {profiles_html}
</div>
</body>
</html>
"""

report_path = OUTPUT_DIR / "pipeline_data_audit.html"
report_path.write_text(html_report, encoding="utf-8")

print()
print("Done.")
print(f"HTML report saved to: {report_path}")
print(f"Figures saved to: {OUTPUT_FIG_DIR}")
