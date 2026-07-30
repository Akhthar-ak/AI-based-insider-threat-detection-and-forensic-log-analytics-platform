from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DB_FILE = PROCESSED_DIR / "security_logs.db"

LOGON_FILE = PROCESSED_DIR / "logon_clean.csv"
FILE_FILE = PROCESSED_DIR / "file_clean.csv"
DEVICE_FILE = PROCESSED_DIR / "device_clean.csv"

IFOREST_RESULTS_FILE = PROCESSED_DIR / "anomaly_results_isolation_forest.csv"
OCSVM_RESULTS_FILE = PROCESSED_DIR / "anomaly_results_one_class_svm.csv"

IFOREST_TIMELINE_FILE = PROCESSED_DIR / "timeline_iforest.csv"
OCSVM_TIMELINE_FILE = PROCESSED_DIR / "timeline_ocsvm.csv"


def safe_read_csv_or_table(csv_path: Path, table_name: str) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    if DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()
        finally:
            conn.close()

    return pd.DataFrame()


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
    )


def build_logon_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "user", "source", "pc", "activity", "filename", "device", "action_description"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "user"]).copy()
    df["user"] = clean_text_series(df["user"])
    df["pc"] = clean_text_series(df.get("pc", ""))
    df["activity"] = clean_text_series(df.get("activity", ""))

    df = df[df["user"] != ""].copy()
    df["source"] = "logon"
    df["filename"] = ""
    df["device"] = ""
    df["action_description"] = df.apply(
        lambda row: f"{row['activity']} on {row['pc']}" if row["pc"] else f"{row['activity']}",
        axis=1
    )

    return df[["date", "user", "source", "pc", "activity", "filename", "device", "action_description"]]


def build_file_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "user", "source", "pc", "activity", "filename", "device", "action_description"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "user"]).copy()
    df["user"] = clean_text_series(df["user"])
    df["pc"] = clean_text_series(df.get("pc", ""))
    df["filename"] = clean_text_series(df.get("filename", ""))
    df["activity"] = clean_text_series(df.get("activity", "file access"))

    df = df[df["user"] != ""].copy()
    df["source"] = "file"
    df["device"] = ""

    def make_desc(row):
        activity = row.get("activity", "file access")
        filename = row.get("filename", "")
        pc = row.get("pc", "")
        if filename and pc:
            return f"{activity} on {filename} from {pc}"
        if filename:
            return f"{activity} on {filename}"
        if pc:
            return f"{activity} from {pc}"
        return str(activity)

    df["action_description"] = df.apply(make_desc, axis=1)
    return df[["date", "user", "source", "pc", "activity", "filename", "device", "action_description"]]


def build_device_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "user", "source", "pc", "activity", "filename", "device", "action_description"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "user"]).copy()
    df["user"] = clean_text_series(df["user"])
    df["pc"] = clean_text_series(df.get("pc", ""))
    df["device"] = clean_text_series(df.get("device", ""))
    df["activity"] = clean_text_series(df.get("activity", "device event"))

    df = df[df["user"] != ""].copy()
    df["source"] = "device"
    df["filename"] = ""

    def make_desc(row):
        activity = row.get("activity", "device event")
        device = row.get("device", "")
        pc = row.get("pc", "")
        if device and pc:
            return f"{activity} for {device} on {pc}"
        if device:
            return f"{activity} for {device}"
        if pc:
            return f"{activity} on {pc}"
        return str(activity)

    df["action_description"] = df.apply(make_desc, axis=1)
    return df[["date", "user", "source", "pc", "activity", "filename", "device", "action_description"]]


def build_timeline(all_events: pd.DataFrame, results_df: pd.DataFrame, output_file: Path, model_name: str):
    if results_df.empty or "user" not in results_df.columns or "prediction" not in results_df.columns:
        empty_df = pd.DataFrame(columns=["model", "date", "user", "source", "pc", "activity", "filename", "device", "action_description"])
        empty_df.to_csv(output_file, index=False)
        print(f"No valid anomaly results for {model_name}. Empty timeline saved: {output_file}")
        return

    suspicious_users = results_df.loc[results_df["prediction"] == -1, "user"].dropna().astype(str).unique()
    timeline = all_events[all_events["user"].astype(str).isin(suspicious_users)].copy()

    if timeline.empty:
        empty_df = pd.DataFrame(columns=["model", "date", "user", "source", "pc", "activity", "filename", "device", "action_description"])
        empty_df.to_csv(output_file, index=False)
        print(f"No suspicious users found for {model_name}. Empty timeline saved: {output_file}")
        return

    timeline = timeline.sort_values(by=["user", "date"])
    timeline["model"] = model_name
    timeline = timeline[["model", "date", "user", "source", "pc", "activity", "filename", "device", "action_description"]]
    timeline.to_csv(output_file, index=False)
    print(f"Forensic timeline created for {model_name}: {output_file}")


def main():
    logon = safe_read_csv_or_table(LOGON_FILE, "logon_logs")
    file_df = safe_read_csv_or_table(FILE_FILE, "file_logs")
    device_df = safe_read_csv_or_table(DEVICE_FILE, "device_logs")

    iforest_results = safe_read_csv_or_table(IFOREST_RESULTS_FILE, "iforest_results")
    ocsvm_results = safe_read_csv_or_table(OCSVM_RESULTS_FILE, "ocsvm_results")

    logon_events = build_logon_events(logon)
    file_events = build_file_events(file_df)
    device_events = build_device_events(device_df)

    all_events = pd.concat([logon_events, file_events, device_events], ignore_index=True)
    if not all_events.empty:
        all_events["date"] = pd.to_datetime(all_events["date"], errors="coerce")
        all_events = all_events.dropna(subset=["date"]).copy()

    build_timeline(all_events, iforest_results, IFOREST_TIMELINE_FILE, "Isolation Forest")
    build_timeline(all_events, ocsvm_results, OCSVM_TIMELINE_FILE, "One-Class SVM")


if __name__ == "__main__":
    main()