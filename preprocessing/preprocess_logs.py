from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "file.csv"
OUT_FILE = ROOT / "data" / "processed" / "file_clean.csv"


def find_column(columns, candidates):
    """
    Find a column name by exact match first, then by partial match.
    """
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


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = find_column(df.columns, ["date", "timestamp", "time"])
    user_col = find_column(df.columns, ["user", "username", "employee"])
    pc_col = find_column(df.columns, ["pc", "computer", "host", "machine"])
    filename_col = find_column(df.columns, ["filename", "file", "path"])
    activity_col = find_column(df.columns, ["activity", "action", "operation", "access"])

    if date_col is None:
        raise ValueError("Could not find a date/timestamp column in file.csv")
    if user_col is None:
        raise ValueError("Could not find a user column in file.csv")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["user"] = df[user_col].astype(str).str.strip()

    if pc_col is not None:
        out["pc"] = df[pc_col].astype(str).str.strip()
    else:
        out["pc"] = ""

    if filename_col is not None:
        out["filename"] = df[filename_col].astype(str).str.strip()
    else:
        out["filename"] = ""

    if activity_col is not None:
        out["activity"] = df[activity_col].astype(str).str.strip()
    else:
        out["activity"] = "file_access"

    out = out.dropna(subset=["date", "user"])
    out["activity"] = out["activity"].replace("nan", "file_access")
    out["filename"] = out["filename"].replace("nan", "")
    out["pc"] = out["pc"].replace("nan", "")

    out = out.sort_values("date")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print("File preprocessing completed successfully")
    print(f"Saved: {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()