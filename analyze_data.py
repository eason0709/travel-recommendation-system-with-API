"""Optional data diagnostics and visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt

from travel_utils import DATA_DIR, FIGURE_DIR, MODEL_FEATURES, ensure_dirs, load_required_csv


def save_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(16, 12))
    image = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(image)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=300)
    plt.close()


def save_distribution_plots(df):
    for feature in MODEL_FEATURES:
        plt.figure(figsize=(6, 4))
        plt.hist(df[feature].dropna(), bins=30)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{feature}_distribution.png", dpi=300)
        plt.close()


def main() -> None:
    ensure_dirs(FIGURE_DIR)
    df = load_required_csv(DATA_DIR / "travel_behavior.csv")
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing Values:")
    print(df.isnull().sum())
    print("Statistical Summary:")
    print(df.describe())
    save_correlation_matrix(df)
    save_distribution_plots(df)
    print("Analyze completed.")


if __name__ == "__main__":
    main()
