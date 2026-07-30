from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DB_FILE = PROCESSED_DIR / "security_logs.db"

FILES_TO_TABLES = {
    "logon_clean.csv": "logon_logs",
    "file_clean.csv": "file_logs",
    "device_clean.csv": "device_logs",
    "features.csv": "user_features",
    "train_features.csv": "train_features",
    "test_features.csv": "test_features",
    "anomaly_results_isolation_forest.csv": "iforest_results",
    "anomaly_results_one_class_svm.csv": "ocsvm_results",
    "timeline_iforest.csv": "timeline_iforest",
    "timeline_ocsvm.csv": "timeline_ocsvm"
}


def main():
    conn = sqlite3.connect(DB_FILE)

    for file_name, table_name in FILES_TO_TABLES.items():
        file_path = PROCESSED_DIR / file_name
        if file_path.exists():
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Loaded {file_name} -> table {table_name}")
        else:
            print(f"Skipped {file_name} (file not found)")

    conn.commit()
    conn.close()
    print(f"Database created/updated: {DB_FILE}")


if __name__ == "__main__":
    main()