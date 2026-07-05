from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_dataset, TARGET_COL

PLOT_DIR = Path("artifacts/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    df = load_dataset()

    df.describe(include="all").to_csv(PLOT_DIR / "summary_statistics.csv")
    df.isna().sum().to_csv(PLOT_DIR / "missing_values.csv")

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df)
    plt.title("Class Distribution - Heart Disease Target")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_distribution.png")
    plt.close()

    # Histograms
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols].hist(figsize=(14, 10))
    plt.suptitle("Numeric Feature Histograms")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "histograms.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png")
    plt.close()

    print(f"EDA completed. Outputs saved under {PLOT_DIR}")


if __name__ == "__main__":
    run_eda()
