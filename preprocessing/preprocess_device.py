from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "device.csv"
OUT_FILE = ROOT / "data" / "processed" / "device_clean.csv"


def find_column(columns, candidates):
    normalized = [c.lower().strip() for c in columns]

    for cand in candidates:
        cand = cand.lower().strip()
        for col in normalized:
            if col == cand:
                return col

    for cand in candidates:
        cand = cand.lower().strip()
        for col in normalized:
            if cand in col:
                return col

    return None


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
    )


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = find_column(df.columns, ["date", "timestamp", "time"])
    user_col = find_column(df.columns, ["user", "username", "employee"])
    pc_col = find_column(df.columns, ["pc", "computer", "host", "machine"])
    device_col = find_column(df.columns, ["device", "device_name", "deviceid", "device_id", "usb", "media", "drive"])
    activity_col = find_column(df.columns, ["activity", "action", "operation", "status", "event"])

    if date_col is None:
        raise ValueError("Could not find a date/timestamp column in device.csv")
    if user_col is None:
        raise ValueError("Could not find a user column in device.csv")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["user"] = clean_text_series(df[user_col])

    if pc_col is not None:
        out["pc"] = clean_text_series(df[pc_col])
    else:
        out["pc"] = ""

    if device_col is not None:
        out["device"] = clean_text_series(df[device_col])
    else:
        out["device"] = ""

    if activity_col is not None:
        out["activity"] = clean_text_series(df[activity_col])
    else:
        out["activity"] = "device_event"

    out = out.dropna(subset=["date", "user"])
    out = out[out["user"].astype(str).str.strip() != ""].copy()

    out["activity"] = out["activity"].replace("", "device_event")
    out["pc"] = out["pc"].replace("", "")
    out["device"] = out["device"].replace("", "")

    out = out.sort_values("date")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print("Device preprocessing completed successfully")
    print(f"Saved: {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()