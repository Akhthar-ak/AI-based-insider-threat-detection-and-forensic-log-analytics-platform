from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

LOGON_FILE = PROCESSED_DIR / "logon_clean.csv"
FILE_FILE = PROCESSED_DIR / "file_clean.csv"
DEVICE_FILE = PROCESSED_DIR / "device_clean.csv"
OUT_FILE = PROCESSED_DIR / "features.csv"


def safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return pd.DataFrame()


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
    )


def safe_nunique(series: pd.Series) -> int:
    if series is None or series.empty:
        return 0
    s = clean_text_series(series)
    s = s[s != ""]
    return int(s.nunique())


def build_logon_features(logon: pd.DataFrame) -> pd.DataFrame:
    if logon.empty:
        return pd.DataFrame(columns=["user", "login_count", "unique_pc", "logon_events", "logoff_events"])

    logon = logon.copy()
    logon["user"] = clean_text_series(logon["user"])
    logon["pc"] = clean_text_series(logon.get("pc", ""))
    logon["activity"] = clean_text_series(logon.get("activity", "")).str.lower()

    logon = logon[logon["user"] != ""].copy()

    total_login = logon.groupby("user").size().reset_index(name="login_count")
    unique_pc = logon.groupby("user")["pc"].apply(safe_nunique).reset_index(name="unique_pc")

    logon_events = (
        logon[logon["activity"].str.contains("logon", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="logon_events")
    )

    logoff_events = (
        logon[logon["activity"].str.contains("logoff", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="logoff_events")
    )

    features = total_login.merge(unique_pc, on="user", how="outer")
    features = features.merge(logon_events, on="user", how="outer")
    features = features.merge(logoff_events, on="user", how="outer")
    return features


def build_file_features(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame(
            columns=["user", "file_access_count", "unique_files_accessed", "open_events", "copy_events", "delete_events"]
        )

    file_df = file_df.copy()
    file_df["user"] = clean_text_series(file_df["user"])
    file_df["filename"] = clean_text_series(file_df.get("filename", ""))
    file_df["activity"] = clean_text_series(file_df.get("activity", "")).str.lower()

    file_df = file_df[file_df["user"] != ""].copy()

    file_access_count = file_df.groupby("user").size().reset_index(name="file_access_count")
    unique_files_accessed = file_df.groupby("user")["filename"].apply(safe_nunique).reset_index(name="unique_files_accessed")

    open_events = (
        file_df[file_df["activity"].str.contains("open", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="open_events")
    )

    copy_events = (
        file_df[file_df["activity"].str.contains("copy", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="copy_events")
    )

    delete_events = (
        file_df[file_df["activity"].str.contains("delete", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="delete_events")
    )

    features = file_access_count.merge(unique_files_accessed, on="user", how="outer")
    features = features.merge(open_events, on="user", how="outer")
    features = features.merge(copy_events, on="user", how="outer")
    features = features.merge(delete_events, on="user", how="outer")
    return features


def build_device_features(device_df: pd.DataFrame) -> pd.DataFrame:
    if device_df.empty:
        return pd.DataFrame(
            columns=["user", "device_event_count", "unique_devices_used", "connect_events", "disconnect_events"]
        )

    device_df = device_df.copy()
    device_df["user"] = clean_text_series(device_df["user"])
    device_df["device"] = clean_text_series(device_df.get("device", ""))
    device_df["activity"] = clean_text_series(device_df.get("activity", "")).str.lower()

    device_df = device_df[device_df["user"] != ""].copy()

    device_event_count = device_df.groupby("user").size().reset_index(name="device_event_count")
    unique_devices_used = device_df.groupby("user")["device"].apply(safe_nunique).reset_index(name="unique_devices_used")

    connect_events = (
        device_df[device_df["activity"].str.contains("connect", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="connect_events")
    )

    disconnect_events = (
        device_df[device_df["activity"].str.contains("disconnect", na=False)]
        .groupby("user")
        .size()
        .reset_index(name="disconnect_events")
    )

    features = device_event_count.merge(unique_devices_used, on="user", how="outer")
    features = features.merge(connect_events, on="user", how="outer")
    features = features.merge(disconnect_events, on="user", how="outer")
    return features


def main():
    logon = safe_read(LOGON_FILE)
    file_df = safe_read(FILE_FILE)
    device_df = safe_read(DEVICE_FILE)

    logon_features = build_logon_features(logon)
    file_features = build_file_features(file_df)
    device_features = build_device_features(device_df)

    features = None

    for part in [logon_features, file_features, device_features]:
        if part.empty:
            continue
        if features is None:
            features = part.copy()
        else:
            features = features.merge(part, on="user", how="outer")

    if features is None or features.empty:
        raise ValueError("No input data available for feature engineering.")

    features = features.fillna(0)

    for col in features.columns:
        if col != "user":
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0).astype(int)

    features = features.sort_values("user")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FILE, index=False)

    print("Feature engineering completed successfully")
    print(f"Saved: {OUT_FILE}")
    print(features.head())
    print("Columns:", list(features.columns))


if __name__ == "__main__":
    main()