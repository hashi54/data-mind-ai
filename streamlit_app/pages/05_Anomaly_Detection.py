import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.data.dataset_manager import dataset_manager
from app.analytics.dynamic_analytics import DynamicAnalyticsEngine
from streamlit_app.components.cards import render_metric_card
from streamlit_app.components.charts import DARK_THEME_LAYOUT

st.set_page_config(page_title="Anomaly Detection | DataMind AI", page_icon="⚡", layout="wide")

st.markdown("## ⚡ Anomaly Detection & Operational Risk Scanner")
st.caption("Auto-detects statistical outliers and unusual patterns in your company data using Rolling Z-Score analysis.")

# ── Load Dataset ─────────────────────────────────────────────────────────────
active_name = dataset_manager.get_active_dataset_name()

if not active_name:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:3rem;">📂</div>
        <div style="font-size:1.3rem; font-weight:700; color:#94a3b8; margin-top:12px;">No Dataset Loaded</div>
        <div style="color:#64748b; margin-top:8px;">Go to <strong>📂 Data Workspace</strong> and upload your company data first.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df = dataset_manager.get_dataset_df(active_name)
roles = DynamicAnalyticsEngine.infer_column_roles(df)

date_col   = roles["date_col"]
metric_col = roles["primary_metric"]

# Sidebar column selectors
with st.sidebar:
    st.markdown("#### 🎛️ Anomaly Settings")
    all_cols      = list(df.columns)
    date_options  = [c for c in all_cols if "date" in c.lower() or "time" in c.lower()] or all_cols
    metric_options = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

    sel_date   = st.selectbox("Date Column (optional)", ["(auto-detect)"] + date_options, key="anom_date")
    sel_metric = st.selectbox("Metric to Analyse", metric_options if metric_options else all_cols, key="anom_metric")
    if sel_date == "(auto-detect)":
        sel_date = date_col

# ── Run Anomaly Detection ─────────────────────────────────────────────────────
with st.spinner("Scanning your dataset for anomalies..."):
    result = DynamicAnalyticsEngine.run_dynamic_anomalies(df, date_col=sel_date, metric_col=sel_metric)

anomalies     = result["anomalies"]
total_anomalies = result["total_anomalies_detected"]
critical_count  = sum(1 for a in anomalies if a.get("severity") == "Critical")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Anomalies Detected", str(total_anomalies), f"In '{sel_metric}'", icon="🚨")
with c2:
    render_metric_card("Critical Deviations", str(critical_count), "Drops > 40%", delta_color="inverse", icon="⚠️")
with c3:
    warning_count = sum(1 for a in anomalies if a.get("severity") == "Warning")
    render_metric_card("Surge Warnings", str(warning_count), "Spikes > 50%", icon="📈")
with c4:
    render_metric_card("Algorithm", "Rolling Z-Score", "14-Day Baseline Window", icon="🧠")

st.divider()

# ── Time Series Chart with Anomaly Markers ────────────────────────────────────
if sel_date and sel_date in df.columns:
    df_ts = df[[sel_date, sel_metric]].copy().dropna()
    df_ts[sel_date] = pd.to_datetime(df_ts[sel_date], errors="coerce")
    df_ts = df_ts.dropna().sort_values(by=sel_date)
    df_daily = df_ts.groupby(pd.Grouper(key=sel_date, freq="D"))[sel_metric].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily[sel_date], y=df_daily[sel_metric],
        name=f"Daily {sel_metric}", mode="lines",
        line=dict(color="#6366f1", width=1.5),
    ))

    if anomalies and "date" in anomalies[0]:
        df_anom = pd.DataFrame([a for a in anomalies if "date" in a and "row" not in a["date"].lower()])
        if not df_anom.empty:
            df_anom["date"] = pd.to_datetime(df_anom["date"], errors="coerce")
            fig.add_trace(go.Scatter(
                x=df_anom["date"], y=df_anom["actual_value"],
                mode="markers", name="Anomaly",
                marker=dict(color="#ef4444", size=11, symbol="diamond", line=dict(color="#fff", width=1.5)),
            ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=f"<b>{sel_metric} — Time Series with Anomaly Flags</b>",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # No date column — just show IQR outlier distribution
    if sel_metric and sel_metric in df.columns:
        fig_box = px.box(df, y=sel_metric, title=f"<b>Outlier Distribution: {sel_metric}</b>", color_discrete_sequence=["#6366f1"])
        fig_box.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

# ── Anomaly Log Table ─────────────────────────────────────────────────────────
st.markdown("### 📋 Anomaly Log & Root Cause Diagnostics")

if anomalies:
    rows = []
    for a in anomalies:
        sev = a.get("severity", "Moderate")
        icon = "🔴" if sev == "Critical" else "🟡" if sev == "Warning" else "🟠"
        rows.append({
            "Period / Row": a.get("date", "—"),
            "Metric": a.get("metric", sel_metric),
            "Actual Value": f"{a['actual_value']:,.2f}",
            "Expected Baseline": f"{a['expected_value']:,.2f}",
            "Deviation": f"{a['deviation_pct']:+0.1f}%",
            "Severity": f"{icon} {sev}",
            "Root Cause Hint": a.get("root_cause_hint", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.success(f"✅ No significant anomalies detected in `{sel_metric}`. Data looks normal.")
