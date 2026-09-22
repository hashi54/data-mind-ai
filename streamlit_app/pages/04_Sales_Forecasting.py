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

st.set_page_config(page_title="Sales Forecasting | DataMind AI", page_icon="🔮", layout="wide")

st.markdown("## 🔮 AI Sales & Revenue Forecasting")
st.caption("Auto-trains on your uploaded data and predicts future trends with 95% confidence intervals.")

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

# Sidebar: allow user to pick columns manually
with st.sidebar:
    st.markdown("#### 🎛️ Forecast Settings")
    available_date_cols   = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "day" in c.lower()] or list(df.columns)
    available_metric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    sel_date   = st.selectbox("Date Column", available_date_cols, index=0 if date_col not in available_date_cols else available_date_cols.index(date_col) if date_col else 0, key="fc_date")
    sel_metric = st.selectbox("Metric to Forecast", available_metric_cols, index=0 if metric_col not in available_metric_cols else available_metric_cols.index(metric_col) if metric_col else 0, key="fc_metric")
    horizon = st.select_slider("Forecast Horizon (days)", options=[7, 14, 30, 60, 90, 180, 365], value=30, key="fc_horizon")

if not sel_date or not sel_metric:
    st.warning("⚠️ Please select a date column and a numeric metric column in the sidebar.")
    st.stop()

if sel_date == sel_metric:
    st.warning(
        f"⚠️ **Column Selection Mismatch:** You selected `{sel_metric}` as both the Date column and the Metric to forecast. "
        "Please select a **time/date column** for Date and a **numeric value column** (like Sales, Amount, Revenue) for Metric.",
        icon="⚠️"
    )
    st.stop()

# ── Run Forecast ─────────────────────────────────────────────────────────────
with st.spinner(f"Training forecast model on `{active_name}` → predicting next {horizon} days..."):
    try:
        result = DynamicAnalyticsEngine.run_dynamic_forecast(df, date_col=sel_date, metric_col=sel_metric, horizon_days=horizon)
    except Exception as e:
        st.warning(
            f"💡 **Forecast Notice:** Could not generate time-series forecast for column `{sel_metric}` with date `{sel_date}`.\n\n"
            f"**Reason:** Continuous time-series data is required. Please pick a numerical value column (e.g. Sales, Amount, Revenue) in the sidebar settings.",
            icon="💡"
        )
        st.stop()

# ── KPI Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card(f"Forecast Total ({horizon}d)", f"{result['total_forecast']:,.2f}", f"{result['growth_pct']:+0.1f}% vs prior period", icon="🔮")
with c2:
    render_metric_card("Model Accuracy (R²)", f"{result['metrics']['r2_score']:.3f}", "Random Forest Regressor", icon="🧠")
with c3:
    render_metric_card("RMSE Error", f"{result['metrics']['rmse']:,.2f}", f"MAE: {result['metrics']['mae']:,.2f}", icon="📐")
with c4:
    render_metric_card("Metric Forecasted", sel_metric, f"Source: {active_name}", icon="📊")

st.divider()

# ── Forecast Chart ────────────────────────────────────────────────────────────
st.markdown(f"### 📈 {sel_metric} — Historical + {horizon}-Day Forecast")

df_hist = pd.DataFrame(result["history_data"]).tail(60)
df_fore = pd.DataFrame(result["forecast_data"])

fig = go.Figure()

# Historical
if "actual_value" in df_hist.columns:
    fig.add_trace(go.Scatter(
        x=df_hist["date"], y=df_hist["actual_value"],
        name="Historical", mode="lines",
        line=dict(color="#6366f1", width=2),
    ))

# Confidence band
fig.add_trace(go.Scatter(
    x=list(df_fore["date"]) + list(df_fore["date"])[::-1],
    y=list(df_fore["upper_bound"]) + list(df_fore["lower_bound"])[::-1],
    fill="toself",
    fillcolor="rgba(168,85,247,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="95% Confidence",
    showlegend=True,
))

# Forecast line
fig.add_trace(go.Scatter(
    x=df_fore["date"], y=df_fore["forecast_value"],
    name="Forecast", mode="lines",
    line=dict(color="#a855f7", width=2.5, dash="dot"),
))

fig.update_layout(**DARK_THEME_LAYOUT, title=f"<b>{sel_metric} — {horizon}-Day Forecast</b>", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ── Forecast Table ────────────────────────────────────────────────────────────
st.markdown("### 📋 Detailed Forecast Table")
df_table = df_fore.copy()
df_table.columns = ["Date", "Forecast Value", "Lower Bound", "Upper Bound"]
st.dataframe(df_table, use_container_width=True)

# Download
csv_bytes = df_table.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Forecast CSV", csv_bytes, f"{active_name}_forecast_{horizon}d.csv", "text/csv")
