import joblib
import numpy as np
from pathlib import Path
import sqlite3
import math
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "data" / "processed" / "security_logs.db"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
    ],
}

st.set_page_config(
    page_title="Insider Threat Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    h1 a, h2 a, h3 a, h4 a {
        display: none !important;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 16px;
        padding: 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        min-height: 110px;
    }

    .metric-icon {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 800;
        flex-shrink: 0;
    }

    .icon-purple { background: rgba(124, 58, 237, 0.18); color: #a78bfa; }
    .icon-red { background: rgba(239, 68, 68, 0.18); color: #f87171; }
    .icon-yellow { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
    .icon-green { background: rgba(34, 197, 94, 0.18); color: #4ade80; }
    .icon-blue { background: rgba(59, 130, 246, 0.18); color: #60a5fa; }

    .metric-title {
        font-size: 14px;
        color: #aab4c4;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.0;
    }

    .metric-subtitle {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
    }

    .timeline-item {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .timeline-meta {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 4px;
    }

    .timeline-action {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .timeline-desc {
        font-size: 13px;
        color: #cbd5e1;
        margin-top: 3px;
    }

    .subtle {
        color: #9ca3af;
        font-size: 13px;
    }

    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 6px;
        white-space: nowrap;
    }

    .tag-logon { background: #123a28; color: #4ade80; }
    .tag-file { background: #3a2f08; color: #fbbf24; }
    .tag-device { background: #0f2741; color: #60a5fa; }
    .tag-logoff { background: #3b1220; color: #fb7185; }

    .event-icon {
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 700;
        margin-right: 12px;
        flex-shrink: 0;
    }

    .icon-logon { background: #123a28; color: #4ade80; }
    .icon-file { background: #3a2f08; color: #fbbf24; }
    .icon-device { background: #0f2741; color: #60a5fa; }
    .icon-logoff { background: #3b1220; color: #fb7185; }

    .page-note {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 6px;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 1.25rem;
        margin-bottom: 0.15rem;
    }

    .section-caption {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-bottom: 0.7rem;
    }

    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-title">Insider Threat Detection</div>
    <div class="app-subtitle">AI-Based Insider Threat Detection and Forensic Log Analytics Platform</div>
    """,
    unsafe_allow_html=True
)

if not DB_FILE.exists():
    st.error("Database not found. Run `python forensic/database_loader.py` first.")
    st.stop()

conn = sqlite3.connect(DB_FILE)

def read_table(table_name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception:
        return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def get_engine_tables(engine_name: str):
    if engine_name == "Engine 1 - Isolation Forest":
        return "iforest_results", "timeline_iforest"
    return "ocsvm_results", "timeline_ocsvm"

def icon_for_source(source: str, activity: str = ""):
    src = str(source).lower()
    act = str(activity).lower()

    if src == "logon" or "logon" in act:
        return "↪", "icon-logon", "tag-logon", "Logon"
    if src == "file" or any(k in act for k in ["file open", "file copy", "file write", "file delete", "copy", "write", "delete", "open"]):
        return "📁", "icon-file", "tag-file", "File"
    if src == "device" or any(k in act for k in ["usb", "device", "connect"]):
        return "🔌", "icon-device", "tag-device", "Device"
    if "logoff" in act:
        return "⏻", "icon-logoff", "tag-logoff", "Logoff"
    return "•", "icon-logon", "tag-logon", "Event"

def compute_range_from_preset(df: pd.DataFrame, preset: str, anchor_mode: str):
    min_dt = pd.Timestamp(df["date"].min()).normalize()
    max_dt = pd.Timestamp(df["date"].max()).normalize()

    if preset == "7 Days":
        delta = pd.Timedelta(days=7)
    elif preset == "15 Days":
        delta = pd.Timedelta(days=15)
    elif preset == "1 Month":
        delta = pd.Timedelta(days=30)
    elif preset == "2 Months":
        delta = pd.Timedelta(days=60)
    elif preset == "3 Months":
        delta = pd.Timedelta(days=90)
    elif preset == "6 Months":
        delta = pd.Timedelta(days=180)
    else:
        delta = pd.Timedelta(days=30)

    if anchor_mode == "Earliest activity":
        start = min_dt
        end = min_dt + delta
        if end > max_dt:
            end = max_dt
    else:
        end = max_dt
        start = max_dt - delta
        if start < min_dt:
            start = min_dt

    return start.date(), end.date()

def set_timeline_page(page_num: int):
    st.session_state["timeline_page"] = page_num

def render_pagination(total_pages: int, current_page: int, key_prefix: str = "timeline"):
    if total_pages <= 1:
        return

    items = []
    if current_page > 1:
        items.append(("prev", current_page - 1))

    visible_pages = {1, total_pages, current_page - 1, current_page, current_page + 1}
    visible_pages = {p for p in visible_pages if 1 <= p <= total_pages}
    visible_pages = sorted(list(visible_pages))

    last_p = None
    for p in visible_pages:
        if last_p is not None and p - last_p > 1:
            items.append(("ellipsis", None))
        items.append(("page", p))
        last_p = p

    if current_page < total_pages:
        items.append(("next", current_page + 1))

    cols = st.columns(len(items))
    for idx, item in enumerate(items):
        kind, value = item
        with cols[idx]:
            if kind == "ellipsis":
                st.markdown("<div style='padding-top:0.45rem; text-align:center; color:#9ca3af;'>...</div>", unsafe_allow_html=True)
            elif kind == "prev":
                st.button("←", key=f"{key_prefix}_prev_{current_page}", use_container_width=True, on_click=set_timeline_page, args=(value,))
            elif kind == "next":
                st.button("→", key=f"{key_prefix}_next_{current_page}", use_container_width=True, on_click=set_timeline_page, args=(value,))
            else:
                st.button(
                    str(value),
                    key=f"{key_prefix}_page_{value}",
                    type="primary" if value == current_page else "secondary",
                    use_container_width=True,
                    on_click=set_timeline_page,
                    args=(value,)
                )

def render_metric_card(title, value, subtitle, icon, icon_class):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon {icon_class}">{icon}</div>
            <div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def format_num(v):
    if pd.isna(v):
        return "-"
    try:
        return f"{int(v):,}"
    except Exception:
        try:
            return f"{float(v):,.4f}"
        except Exception:
            return str(v)

def add_sl_no(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy().reset_index(drop=True)
    out.insert(0, "Sl No", range(1, len(out) + 1))
    return out

def search_filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    q = str(query).strip().lower()
    if not q:
        return df
    mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False, regex=False))
    return df[mask.any(axis=1)].copy()

def enrich_results_with_features(results: pd.DataFrame) -> pd.DataFrame:
    results = normalize_columns(results)
    if results.empty or "user" not in results.columns:
        return results

    features = normalize_columns(read_table("user_features"))
    if features.empty or "user" not in features.columns:
        return results

    device_cols = [
        "device_event_count",
        "unique_devices_used",
        "connect_events",
        "disconnect_events",
    ]

    keep_cols = ["user"] + [c for c in device_cols if c in features.columns]
    if len(keep_cols) == 1:
        return results

    features = features[keep_cols].copy()
    merged = results.merge(features, on="user", how="left", suffixes=("", "_feat"))

    for col in device_cols:
        feat_col = f"{col}_feat"
        if feat_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[feat_col])
            else:
                merged[col] = merged[feat_col]
            merged = merged.drop(columns=[feat_col])

    return merged

def build_results_view(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if df.empty:
        return df

    view = df.copy()

    preferred = [
        "user",
        "login_count",
        "unique_pc",
        "logon_events",
        "logoff_events",
        "file_access_count",
        "unique_files_accessed",
        "open_events",
        "copy_events",
        "delete_events",
        "device_event_count",
        "unique_devices_used",
        "connect_events",
        "disconnect_events",
        "anomaly_score",
        "prediction",
    ]
    cols = [c for c in preferred if c in view.columns]
    out = view[cols].copy()

    rename_map = {
        "user": "User",
        "login_count": "Login Count",
        "unique_pc": "Unique PC",
        "logon_events": "Logon Events",
        "logoff_events": "Logoff Events",
        "file_access_count": "File Access",
        "unique_files_accessed": "Unique Files",
        "open_events": "Open Events",
        "copy_events": "Copy Events",
        "delete_events": "Delete Events",
        "device_event_count": "Device Events",
        "unique_devices_used": "Unique Devices",
        "connect_events": "Connect Events",
        "disconnect_events": "Disconnect Events",
        "anomaly_score": "Anomaly Score",
        "prediction": "Prediction"
    }
    out = out.rename(columns=rename_map)

    for col in out.columns:
        if col == "Prediction":
            out[col] = out[col].replace({1: "Normal", -1: "Suspicious"})
        elif col == "Anomaly Score":
            out[col] = out[col].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")
        elif col not in ["User"]:
            out[col] = out[col].apply(format_num)

    return out

def build_disagreement_view(iforest_df: pd.DataFrame, ocsvm_df: pd.DataFrame) -> pd.DataFrame:
    if iforest_df.empty or ocsvm_df.empty:
        return pd.DataFrame()

    merged = iforest_df.merge(ocsvm_df, on="user", suffixes=("_iforest", "_svm"))
    disagreement = merged[merged["prediction_iforest"] != merged["prediction_svm"]].copy()

    if disagreement.empty:
        return pd.DataFrame()

    out = disagreement[[
        "user",
        "prediction_iforest",
        "prediction_svm",
        "anomaly_score_iforest",
        "anomaly_score_svm"
    ]].copy()

    out = out.rename(columns={
        "user": "User",
        "prediction_iforest": "Engine 1 Prediction",
        "prediction_svm": "Engine 2 Prediction",
        "anomaly_score_iforest": "Engine 1 Score",
        "anomaly_score_svm": "Engine 2 Score",
    })

    out["Engine 1 Prediction"] = out["Engine 1 Prediction"].replace({1: "Normal", -1: "Suspicious"})
    out["Engine 2 Prediction"] = out["Engine 2 Prediction"].replace({1: "Normal", -1: "Suspicious"})
    out["Engine 1 Score"] = out["Engine 1 Score"].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")
    out["Engine 2 Score"] = out["Engine 2 Score"].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")

    return out

def determine_risk_score(anomaly_score, total_events: int, file_count: int):
    score_value = None
    if anomaly_score is not None and pd.notna(anomaly_score):
        try:
            score_value = float(anomaly_score)
        except Exception:
            score_value = None

    risk_score = "LOW"

    if score_value is not None:
        if score_value <= -0.60:
            risk_score = "HIGH"
        elif score_value <= -0.25:
            risk_score = "MEDIUM"

    if total_events > 120 or file_count > 40:
        risk_score = "HIGH"

    return risk_score, score_value
def load_prediction_model():

    try:
        return joblib.load(ROOT / "models" / "isolation_forest.pkl")

    except Exception:
        return joblib.load(ROOT / "models" / "anomaly_model.pkl")

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📊 Anomaly Dashboard",
        "🕒 Forensic Timeline",
        "🎯 Threat Simulator"
    ]
)

st.sidebar.markdown("### 🛡️ Detection Engine")
engine_option = st.sidebar.selectbox(
    "Select engine",
    ["🟢 Engine 1 - Isolation Forest", "🟣 Engine 2 - One-Class SVM"]
)

if "Isolation Forest" in engine_option:
    results_table, timeline_table = "iforest_results", "timeline_iforest"
    engine_name = "Isolation Forest"
else:
    results_table, timeline_table = "ocsvm_results", "timeline_ocsvm"
    engine_name = "One-Class SVM"

# ================================
# ANOMALY DASHBOARD
# ================================
if page == "📊 Anomaly Dashboard":
    results = enrich_results_with_features(read_table(results_table))
    if results.empty:
        st.warning("No detection results found. Run detection first.")
        st.stop()

    if "model" in results.columns:
        results = results.drop(columns=["model"], errors="ignore")

    total_users = len(results)
    suspicious_count = int((results["prediction"] == -1).sum()) if "prediction" in results.columns else 0
    anomaly_rate = (suspicious_count / total_users * 100) if total_users else 0

    iforest_df = normalize_columns(read_table("iforest_results"))
    ocsvm_df = normalize_columns(read_table("ocsvm_results"))
    consistency = 0.0
    disagreement_display = build_disagreement_view(iforest_df, ocsvm_df)

    if not iforest_df.empty and not ocsvm_df.empty and "user" in iforest_df.columns and "user" in ocsvm_df.columns:
        merged = iforest_df.merge(ocsvm_df, on="user", suffixes=("_iforest", "_svm"))
        if not merged.empty:
            same_predictions = merged[merged["prediction_iforest"] == merged["prediction_svm"]]
            consistency = (len(same_predictions) / len(merged)) * 100

    st.markdown("### Detection Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Users", f"{total_users}", "All analyzed users", "👥", "icon-purple")
    with c2:
        render_metric_card("Suspicious Users", f"{suspicious_count}", "Flagged as anomalous", "🛑", "icon-red")
    with c3:
        render_metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%", "Suspicious / Total Users", "📈", "icon-yellow")
    with c4:
        render_metric_card("Detection Consistency", f"{consistency:.2f}%", "Between both engines", "🛡️", "icon-green")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Suspicious vs Normal Users")
        label_df = pd.DataFrame({
            "Status": ["Normal", "Suspicious"],
            "Count": [
                int((results["prediction"] == 1).sum()) if "prediction" in results.columns else 0,
                int((results["prediction"] == -1).sum()) if "prediction" in results.columns else 0
            ]
        })
        fig = px.pie(
            label_df,
            names="Status",
            values="Count",
            hole=0.64,
            color="Status",
            color_discrete_map={
                "Normal": "#22c55e",
                "Suspicious": "#ef4444"
            }
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=320,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        st.markdown("#### 🔵 Anomaly Score Distribution")
        if "anomaly_score" in results.columns:
            score_df = results.copy()
            score_df["anomaly_score"] = pd.to_numeric(score_df["anomaly_score"], errors="coerce")
            score_df = score_df.dropna(subset=["anomaly_score"])

            hist = px.histogram(
                score_df,
                x="anomaly_score",
                nbins=28,
                color_discrete_sequence=["#4f83ff"]
            )
            hist.update_traces(marker_line_color="#1e40af", marker_line_width=0.6)
            hist.update_layout(
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=320
            )
            st.plotly_chart(hist, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Anomaly score column not available.")

    st.markdown("#### 🟣 Top 10 Users by Event Volume")
    event_cols = [c for c in [
        "login_count",
        "file_access_count",
        "logon_events",
        "logoff_events",
        "open_events",
        "copy_events",
        "delete_events",
        "device_event_count",
        "connect_events",
        "disconnect_events"
    ] if c in results.columns]

    if event_cols and "user" in results.columns:
        plot_df = results.copy()
        plot_df["total_events"] = plot_df[event_cols].sum(axis=1)
        top_users = plot_df.sort_values("total_events", ascending=False).head(10)
        bar = px.bar(
            top_users,
            x="total_events",
            y="user",
            orientation="h",
            color="total_events",
            color_continuous_scale=[[0, "#6d28d9"], [1, "#ec4899"]],
            text="total_events"
        )
        bar.update_traces(textposition="outside", cliponaxis=False)
        bar.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=360,
            xaxis_title="Total Events",
            yaxis_title="User",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.info("No event count columns available for the bar chart.")

    st.markdown("#### 👥 All Users")
    st.caption("All analyzed users with behavior features and engine output.")
    all_users_search = st.text_input(
        "Search All Users",
        placeholder="Type any value to filter users",
        key="all_users_search"
    )
    all_users_view = add_sl_no(build_results_view(results))
    all_users_view = search_filter_df(all_users_view, all_users_search)
    st.dataframe(
        all_users_view,
        use_container_width=True,
        hide_index=True,
        height=330
    )

    st.markdown(f"#### 🛑 Suspicious Users ({suspicious_count})")
    st.caption("Users flagged as suspicious by the selected engine.")
    suspicious_search = st.text_input(
        "Search Suspicious Users",
        placeholder="Type any value to filter suspicious users",
        key="suspicious_search"
    )
    suspicious_view = results[results["prediction"] == -1].copy() if "prediction" in results.columns else results.copy()
    suspicious_view = add_sl_no(build_results_view(suspicious_view))
    suspicious_view = search_filter_df(suspicious_view, suspicious_search)
    if suspicious_view.empty:
        st.info("No suspicious users found.")
    else:
        st.dataframe(
            suspicious_view,
            use_container_width=True,
            hide_index=True,
            height=330
        )

    st.markdown("#### ⚠️ Analyst Attention Required")
    st.caption("Users where both engines disagree and need manual review.")
    disagreement_search = st.text_input(
        "Search Disagreement Users",
        placeholder="Type any value to filter disagreements",
        key="disagreement_search"
    )
    disagreement_display = add_sl_no(disagreement_display)
    disagreement_display = search_filter_df(disagreement_display, disagreement_search)
    if disagreement_display.empty:
        st.info("No disagreement detected between the detection engines.")
    else:
        st.dataframe(
            disagreement_display,
            use_container_width=True,
            hide_index=True,
            height=260
        )

# ================================
# FORENSIC TIMELINE
# ================================
elif page == "🕒 Forensic Timeline":
    timeline = normalize_columns(read_table(timeline_table))
    if timeline.empty:
        st.warning("No timeline data found for the selected engine.")
        st.stop()

    if "date" not in timeline.columns or "user" not in timeline.columns:
        st.error("Timeline data is missing required columns.")
        st.stop()

    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline = timeline.dropna(subset=["date"]).copy()

    if timeline.empty:
        st.warning("Timeline contains no valid dates.")
        st.stop()

    if "source" not in timeline.columns:
        timeline["source"] = "logon"
    if "activity" not in timeline.columns:
        timeline["activity"] = "event"
    if "action_description" not in timeline.columns:
        timeline["action_description"] = timeline["activity"].astype(str)
    if "pc" not in timeline.columns:
        timeline["pc"] = ""
    if "filename" not in timeline.columns:
        timeline["filename"] = ""

    st.markdown("### Forensic Timeline Investigation")
    st.caption("Explore chronological activities of the selected user.")

    st.sidebar.markdown("### 👤 User Selection")
    users = sorted(timeline["user"].dropna().astype(str).unique().tolist())
    if not users:
        st.warning("No suspicious users found for the selected engine.")
        st.stop()

    selected_user = st.sidebar.selectbox("Select user", users)

    st.sidebar.markdown("### ⏱️ Timeline Direction")
    anchor_mode = st.sidebar.radio(
        "Select activity window from",
        ["Newest activity", "Earliest activity"],
        index=0
    )

    st.sidebar.markdown("### 🗓️ Time Range Presets")
    preset = st.sidebar.radio(
        "Choose a preset",
        ["7 Days", "15 Days", "1 Month", "2 Months", "3 Months", "6 Months", "Custom Range"],
        index=2
    )

    min_date = timeline["date"].min().date()
    max_date = timeline["date"].max().date()

    if preset == "Custom Range":
        with st.sidebar.form("timeline_filter_form"):
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            apply_filters = st.form_submit_button("Apply Filters")
    else:
        start_date, end_date = compute_range_from_preset(timeline, preset, anchor_mode)
        apply_filters = True

    st.sidebar.markdown("### 📄 Entries Per Page")
    rows_per_page = st.sidebar.selectbox("Show", [5, 8, 10, 15, 20], index=2)

    user_timeline = timeline[timeline["user"].astype(str) == str(selected_user)].copy()
    if user_timeline.empty:
        st.warning("No events found for the selected user.")
        st.stop()

    if "timeline_page" not in st.session_state:
        st.session_state["timeline_page"] = 1

    if apply_filters:
        if start_date > end_date:
            st.error("Start date must be before or equal to end date.")
            st.stop()

        filtered = user_timeline[
            (user_timeline["date"].dt.date >= start_date) &
            (user_timeline["date"].dt.date <= end_date)
        ].copy()

        if filtered.empty:
            st.warning("No activity found for the selected date range.")
            st.stop()

        filtered = filtered.sort_values("date").reset_index(drop=True)

        total_events = len(filtered)
        logon_count = len(filtered[filtered["source"].astype(str).str.lower() == "logon"])
        file_count = len(filtered[filtered["source"].astype(str).str.lower() == "file"])
        device_count = len(filtered[filtered["source"].astype(str).str.lower() == "device"])
        logoff_count = len(filtered[filtered["activity"].astype(str).str.lower().str.contains("logoff", na=False)])

        selected_results = enrich_results_with_features(read_table(results_table))
        anomaly_score = None
        if not selected_results.empty and "user" in selected_results.columns and "anomaly_score" in selected_results.columns:
            row = selected_results[selected_results["user"].astype(str) == str(selected_user)]
            if not row.empty:
                anomaly_score = row.iloc[0]["anomaly_score"]

        risk_score, score_value = determine_risk_score(anomaly_score, total_events, file_count)

        top1, top2, top3, top4, top5 = st.columns(5)
        with top1:
            render_metric_card("Total Events", f"{total_events}", "In selected range", "⚡", "icon-purple")
        with top2:
            render_metric_card("Logon Activities", f"{logon_count}", "Authentication events", "↪", "icon-green")
        with top3:
            render_metric_card("File Activities", f"{file_count}", "File operations", "📁", "icon-yellow")
        with top4:
            render_metric_card("Device Activities", f"{device_count}", "USB / device events", "🔌", "icon-blue")
        with top5:
            render_metric_card("Logoff Activities", f"{logoff_count}", "Session exits", "⏻", "icon-red")

        left_col, right_col = st.columns([2.25, 1.15])

        with left_col:
            st.markdown("#### 🕘 Activity Timeline")
            st.caption("Chronological view of user activities.")

            total_pages = max(1, math.ceil(len(filtered) / rows_per_page))
            if st.session_state["timeline_page"] > total_pages:
                st.session_state["timeline_page"] = total_pages

            current_page = st.session_state["timeline_page"]
            start_idx = (current_page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            page_df = filtered.iloc[start_idx:end_idx].copy()

            for _, row in page_df.iterrows():
                icon, icon_class, tag_class, tag_text = icon_for_source(row.get("source", ""), row.get("activity", ""))
                pc = str(row.get("pc", "")).strip()
                filename = str(row.get("filename", "")).strip()
                file_text = filename if filename and filename.lower() != "nan" else "-"
                pc_text = pc if pc and pc.lower() != "nan" else "-"

                st.markdown(
                    f"""
                    <div class="timeline-item">
                        <div style="display:flex; align-items:flex-start; gap:12px;">
                            <div class="event-icon {icon_class}">{icon}</div>
                            <div style="flex:1;">
                                <div class="timeline-meta">{row['date']}</div>
                                <div class="timeline-action">{row.get('activity', 'Event')}</div>
                                <div class="timeline-desc">{row.get('action_description', '')}</div>
                                <div class="subtle">PC/Device: {pc_text} | File: {file_text}</div>
                            </div>
                            <span class="tag {tag_class}">{tag_text}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<div class='page-note'>Showing {start_idx + 1} to {min(end_idx, len(filtered))} of {len(filtered)} entries</div>",
                unsafe_allow_html=True
            )
            render_pagination(total_pages, current_page, key_prefix=f"{selected_user}_{preset}_{anchor_mode}")

        with right_col:
            st.markdown("#### 📊 Activity Distribution")
            st.caption("Breakdown of activities in selected range.")

            dist_df = pd.DataFrame({
                "Activity": ["Logon", "File", "Device", "Logoff"],
                "Count": [logon_count, file_count, device_count, logoff_count]
            })

            fig = px.pie(
                dist_df,
                names="Activity",
                values="Count",
                hole=0.64,
                color="Activity",
                color_discrete_map={
                    "Logon": "#22c55e",
                    "File": "#fbbf24",
                    "Device": "#60a5fa",
                    "Logoff": "#ef4444"
                }
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=320,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            st.markdown("#### 🗂️ Top Accessed Files")
            st.caption("Most frequently accessed files in the selected range.")
            file_search = st.text_input(
                "Search Top Accessed Files",
                placeholder="Type any value to filter files",
                key="file_search"
            )

            file_events = filtered[filtered["source"].astype(str).str.lower() == "file"].copy()
            if not file_events.empty and "filename" in file_events.columns:
                top_files = (
                    file_events["filename"]
                    .fillna("")
                    .astype(str)
                    .replace("", "-")
                    .value_counts()
                    .reset_index()
                )
                top_files.columns = ["File Path", "Access Count"]
                top_files = add_sl_no(top_files)
                top_files = search_filter_df(top_files, file_search)

                st.dataframe(
                    top_files,
                    use_container_width=True,
                    hide_index=True,
                    height=260
                )
            else:
                st.info("No file activity found in the selected range.")

            st.markdown("#### 🧭 Risk Summary")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon {'icon-red' if risk_score == 'HIGH' else 'icon-yellow' if risk_score == 'MEDIUM' else 'icon-green'}">⚠️</div>
                    <div>
                        <div class="metric-title">Selected User Risk</div>
                        <div class="metric-value">{risk_score}</div>
                        <div class="metric-subtitle">Based on anomaly score and activity</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if score_value is not None and pd.notna(score_value):
                st.markdown(f"<div class='page-note'><b>Anomaly Score:</b> {score_value:.4f}</div>", unsafe_allow_html=True)

        st.markdown("#### 📄 Raw Event Data")
        raw_search = st.text_input(
            "Search Raw Event Data",
            placeholder="Type any value to filter events",
            key="raw_search"
        )

        raw_cols = [c for c in ["date", "user", "source", "activity", "action_description", "pc", "filename"] if c in filtered.columns]
        raw_df = filtered[raw_cols].copy()

        def clean_value(value, placeholder="—"):
            if pd.isna(value):
                return placeholder
            text = str(value).strip()
            if not text or text.lower() == "nan":
                return placeholder
            return text

        if "pc" in raw_df.columns:
            raw_df["pc"] = raw_df["pc"].apply(clean_value)
        if "filename" in raw_df.columns:
            raw_df["filename"] = raw_df["filename"].apply(clean_value)

        raw_df = raw_df.rename(columns={
            "date": "Date / Time",
            "user": "User",
            "source": "Source",
            "activity": "Activity",
            "action_description": "Details",
            "pc": "PC / Device",
            "filename": "File Path"
        })

        raw_df = add_sl_no(raw_df)
        raw_df = search_filter_df(raw_df, raw_search)

        st.dataframe(
            raw_df,
            use_container_width=True,
            hide_index=True,
            height=340
        )

elif page == "🎯 Threat Simulator":

    st.markdown("## 🎯 Threat Prediction Simulator")
    st.caption(
        "Enter feature values manually and predict whether the behavior is Normal or Suspicious."
    )

    model = load_prediction_model()

    col1, col2 = st.columns(2)

    with col1:
        login_count = st.slider("Login Count", 0, 500, 50)
        unique_pc = st.slider("Unique PC", 0, 20, 2)
        logon_events = st.slider("Logon Events", 0, 500, 50)
        logoff_events = st.slider("Logoff Events", 0, 500, 50)
        file_access_count = st.slider("File Access Count", 0, 5000, 100)

    with col2:
        unique_files_accessed = st.slider("Unique Files Accessed", 0, 2000, 20)
        open_events = st.slider("Open Events", 0, 5000, 100)
        copy_events = st.slider("Copy Events", 0, 1000, 10)
        delete_events = st.slider("Delete Events", 0, 1000, 5)

        device_event_count = st.slider("Device Event Count", 0, 500, 0)
        unique_devices_used = st.slider("Unique Devices Used", 0, 50, 0)
        connect_events = st.slider("Connect Events", 0, 500, 0)
        disconnect_events = st.slider("Disconnect Events", 0, 500, 0)

    if st.button("🚀 Predict Threat", use_container_width=True):

        feature_vector = pd.DataFrame([{
            "login_count": login_count,
            "unique_pc": unique_pc,
            "logon_events": logon_events,
            "logoff_events": logoff_events,
            "file_access_count": file_access_count,
            "unique_files_accessed": unique_files_accessed,
            "open_events": open_events,
            "copy_events": copy_events,
            "delete_events": delete_events,
            "device_event_count": device_event_count,
            "unique_devices_used": unique_devices_used,
            "connect_events": connect_events,
            "disconnect_events": disconnect_events
        }])

        if hasattr(model, "feature_names_in_"):

            expected_cols = list(model.feature_names_in_)

            for col in expected_cols:
                if col not in feature_vector.columns:
                    feature_vector[col] = 0

            feature_vector = feature_vector[expected_cols]

        prediction = model.predict(feature_vector)[0]

        try:
            raw_score = float(
                model.decision_function(feature_vector)[0]
            )
        except Exception:
            raw_score = 0.0

        score_display = float(np.tanh(raw_score))

        # DEBUG
        st.write("Prediction =", prediction)
        st.write("Raw Score =", raw_score)
        st.write("Display Score =", score_display)

        if prediction == -1:
            st.error("⚠️ Suspicious User Detected")
        else:
            st.success("✅ Normal User")

        st.metric(
            "Prediction",
            "Suspicious" if prediction == -1 else "Normal"
        )

        st.metric(
            "Anomaly Score",
            f"{score_display:.4f}"
        )


conn.close()