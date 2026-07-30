from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "features.csv"
TRAIN_FILE = PROCESSED_DIR / "train_features.csv"
TEST_FILE = PROCESSED_DIR / "test_features.csv"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "user" not in df.columns:
        raise ValueError("features.csv must contain a 'user' column")

    feature_cols = [c for c in df.columns if c != "user"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("Train/test split completed")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {TEST_FILE}")

if __name__ == "__main__":
    main()