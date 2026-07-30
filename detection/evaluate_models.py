from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

IFOREST_RESULTS_FILE = PROCESSED_DIR / "anomaly_results_isolation_forest.csv"
OCSVM_RESULTS_FILE = PROCESSED_DIR / "anomaly_results_one_class_svm.csv"
METRICS_FILE = PROCESSED_DIR / "model_metrics.csv"

GROUND_TRUTH_CANDIDATES = [
    RAW_DIR / "ground_truth.csv",
    PROCESSED_DIR / "ground_truth.csv"
]

def load_ground_truth():
    for path in GROUND_TRUTH_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            if {"user", "true_label"}.issubset(df.columns):
                df = df[["user", "true_label"]].copy()
                df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce").fillna(0).astype(int)
                return df, path
    return None, None

def read_results(path: Path, model_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "user" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"{path.name} must contain 'user' and 'prediction' columns")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(1).astype(int)
    df["pred_label"] = (df["prediction"] == -1).astype(int)  # 1 = suspicious, 0 = normal
    df["model"] = model_name
    return df

def compute_metrics(results_df: pd.DataFrame, model_name: str, gt_df: pd.DataFrame | None):
    total_users = len(results_df)
    suspicious_users = int((results_df["prediction"] == -1).sum())
    anomaly_rate = suspicious_users / total_users if total_users else 0

    metrics = {
        "model": model_name,
        "total_users": total_users,
        "suspicious_users": suspicious_users,
        "anomaly_rate": anomaly_rate,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "ground_truth_used": False,
        "agreement_rate": np.nan
    }

    if gt_df is not None:
        merged = results_df.merge(gt_df, on="user", how="inner")
        if not merged.empty:
            y_true = merged["true_label"].astype(int)
            y_pred = merged["pred_label"].astype(int)

            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
            metrics["ground_truth_used"] = True

    return metrics

def main():
    gt_df, gt_path = load_ground_truth()

    iforest_df = read_results(IFOREST_RESULTS_FILE, "Isolation Forest")
    ocsvm_df = read_results(OCSVM_RESULTS_FILE, "One-Class SVM")

    iforest_metrics = compute_metrics(iforest_df, "Isolation Forest", gt_df)
    ocsvm_metrics = compute_metrics(ocsvm_df, "One-Class SVM", gt_df)

    # Model agreement rate (useful even if no ground truth exists)
    merged = iforest_df[["user", "prediction"]].merge(
        ocsvm_df[["user", "prediction"]],
        on="user",
        suffixes=("_iforest", "_ocsvm"),
        how="inner"
    )
    if not merged.empty:
        agreement_rate = (merged["prediction_iforest"] == merged["prediction_ocsvm"]).mean()
    else:
        agreement_rate = np.nan

    iforest_metrics["agreement_rate"] = agreement_rate
    ocsvm_metrics["agreement_rate"] = agreement_rate

    metrics_df = pd.DataFrame([iforest_metrics, ocsvm_metrics])
    metrics_df.to_csv(METRICS_FILE, index=False)

    print("Model evaluation completed")
    if gt_path:
        print(f"Ground truth used: {gt_path}")
    else:
        print("No ground truth file found. Accuracy/precision/recall/F1 left blank.")
    print(f"Saved: {METRICS_FILE}")

if __name__ == "__main__":
    main()