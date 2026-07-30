from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TRAIN_FILE = PROCESSED_DIR / "train_features.csv"

IFOREST_MODEL_FILE = MODELS_DIR / "isolation_forest.pkl"
OCSVM_MODEL_FILE = MODELS_DIR / "one_class_svm.pkl"

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE)

    if "user" not in df.columns:
        raise ValueError("train_features.csv must contain a 'user' column")

    feature_cols = [c for c in df.columns if c != "user"]
    X_train = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    isolation_forest_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=0.08,
            random_state=42
        ))
    ])

    one_class_svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", OneClassSVM(
            kernel="rbf",
            nu=0.08,
            gamma="scale"
        ))
    ])

    isolation_forest_pipeline.fit(X_train)
    one_class_svm_pipeline.fit(X_train)

    joblib.dump(isolation_forest_pipeline, IFOREST_MODEL_FILE)
    joblib.dump(one_class_svm_pipeline, OCSVM_MODEL_FILE)

    print("Model training completed")
    print(f"Saved: {IFOREST_MODEL_FILE}")
    print(f"Saved: {OCSVM_MODEL_FILE}")

if __name__ == "__main__":
    main()