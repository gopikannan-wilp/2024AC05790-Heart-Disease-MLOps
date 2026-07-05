from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "heart_disease_uci.csv"

def download_with_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    # UCI target column is commonly named "num"; convert to binary target.
    target_col = "num" if "num" in df.columns else df.columns[-1]
    df["target"] = (df[target_col] > 0).astype(int)
    if target_col != "target":
        df = df.drop(columns=[target_col])
    df.to_csv(OUT, index=False)
    print(f"Downloaded dataset to {OUT}")

if __name__ == "__main__":
    download_with_ucimlrepo()
