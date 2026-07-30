from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FEATURES_FILE = PROCESSED_DIR / "features.csv"
TEST_FEATURES_FILE = PROCESSED_DIR / "test_features.csv"
TRAIN_FEATURES_FILE = PROCESSED_DIR / "train_features.csv"

IFOREST_OUT = PROCESSED_DIR / "anomaly_results_isolation_forest.csv"
OCSVM_OUT = PROCESSED_DIR / "anomaly_results_one_class_svm.csv"

# Allowed values: "test", "train", "all"
PREDICT_ON = "test"


def load_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
    )


def load_dataframe_by_mode(mode: str) -> pd.DataFrame:
    mode = mode.lower().strip()

    if mode == "test" and TEST_FEATURES_FILE.exists():
        df = pd.read_csv(TEST_FEATURES_FILE)
    elif mode == "train" and TRAIN_FEATURES_FILE.exists():
        df = pd.read_csv(TRAIN_FEATURES_FILE)
    elif FEATURES_FILE.exists():
        df = pd.read_csv(FEATURES_FILE)
    elif TEST_FEATURES_FILE.exists():
        df = pd.read_csv(TEST_FEATURES_FILE)
    elif TRAIN_FEATURES_FILE.exists():
        df = pd.read_csv(TRAIN_FEATURES_FILE)
    else:
        raise FileNotFoundError(
            "No feature file found. Expected features.csv, test_features.csv, or train_features.csv"
        )

    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_model(candidates):
    path = load_first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Model file not found. Checked:\n" + "\n".join(str(p) for p in candidates)
        )
    return joblib.load(path), path


def align_features_to_model(df: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "user" not in df.columns:
        raise ValueError("features file must contain a 'user' column")

    df = df.copy()
    df["user"] = clean_text_series(df["user"])

    valid_mask = df["user"] != ""
    df = df.loc[valid_mask].copy()

    candidate_cols = [c for c in df.columns if c != "user"]

    if hasattr(model, "feature_names_in_"):
        expected = [c for c in list(model.feature_names_in_) if c in df.columns]
        if expected:
            candidate_cols = expected

    X = df[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, X


def normalize_to_minus1_1(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)

    finite_mask = np.isfinite(scores)

    if not finite_mask.any():
        return np.zeros_like(scores)

    finite_scores = scores[finite_mask]

    max_abs = np.max(np.abs(finite_scores))

    if max_abs == 0:
        normalized = np.zeros_like(scores)
    else:
        normalized = scores / max_abs

    normalized = np.clip(normalized, -1, 1)

    normalized[~finite_mask] = 0

    return normalized


def score_isolation_forest(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_df, X = align_features_to_model(df, model)

    predictions = model.predict(X)

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X)
    else:
        raw_scores = np.zeros(len(X), dtype=float)

    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)

    result = feature_df.copy()
    result["prediction"] = predictions.astype(int)
    result["anomaly_score_raw"] = raw_scores
    result["anomaly_score"] = raw_scores

    result = result.sort_values(by=["anomaly_score", "user"], ascending=[True, True]).reset_index(drop=True)
    return result


def score_one_class_svm(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_df, X = align_features_to_model(df, model)

    predictions = model.predict(X)

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
    else:
        raw_scores = np.zeros(len(X), dtype=float)

    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)

    # Convert raw SVM boundary distances to a dashboard-friendly [-1, 1] scale
    display_scores = normalize_to_minus1_1(raw_scores)

    result = feature_df.copy()
    result["prediction"] = predictions.astype(int)
    result["anomaly_score_raw"] = raw_scores
    result["anomaly_score"] = display_scores

    result = result.sort_values(by=["anomaly_score", "user"], ascending=[True, True]).reset_index(drop=True)
    return result


def print_summary(name: str, result_df: pd.DataFrame):
    total = len(result_df)
    suspicious = int((result_df["prediction"] == -1).sum()) if "prediction" in result_df.columns else 0
    normal = int((result_df["prediction"] == 1).sum()) if "prediction" in result_df.columns else 0
    print(f"\n{name}")
    print(f"Total rows: {total}")
    print(f"Normal: {normal}")
    print(f"Suspicious: {suspicious}")
    print(result_df.head())


def main():
    df = load_dataframe_by_mode(PREDICT_ON)

    iforest_model, iforest_path = load_model([
        MODELS_DIR / "isolation_forest.pkl",
        MODELS_DIR / "isolation_forest_model.pkl",
        MODELS_DIR / "iforest_model.pkl",
        PROCESSED_DIR / "isolation_forest.pkl",
        PROCESSED_DIR / "iforest_model.pkl",
    ])

    ocsvm_model, ocsvm_path = load_model([
        MODELS_DIR / "one_class_svm.pkl",
        MODELS_DIR / "one_class_svm_model.pkl",
        MODELS_DIR / "ocsvm_model.pkl",
        PROCESSED_DIR / "one_class_svm.pkl",
        PROCESSED_DIR / "ocsvm_model.pkl",
    ])

    print(f"Using Isolation Forest model: {iforest_path}")
    print(f"Using One-Class SVM model: {ocsvm_path}")
    print(f"Scoring mode: {PREDICT_ON}")

    iforest_results = score_isolation_forest(df, iforest_model)
    ocsvm_results = score_one_class_svm(df, ocsvm_model)

    IFOREST_OUT.parent.mkdir(parents=True, exist_ok=True)
    iforest_results.to_csv(IFOREST_OUT, index=False)
    ocsvm_results.to_csv(OCSVM_OUT, index=False)

    print_summary("Isolation Forest Results", iforest_results)
    print_summary("One-Class SVM Results", ocsvm_results)

    print(f"\nSaved: {IFOREST_OUT}")
    print(f"Saved: {OCSVM_OUT}")
    print("\nNotes:")
    print(" - Isolation Forest anomaly_score is the model's raw score.")
    print(" - One-Class SVM anomaly_score is normalized to [-1, 1] for easier display.")
    print(" - anomaly_score_raw keeps the original model output for both engines.")


if __name__ == "__main__":
    main()