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

st.set_page_config(page_title="Executive Dashboard | DataMind AI", page_icon="📈", layout="wide")


def _no_data_banner():
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:3.5rem;">📂</div>
        <div style="font-size:1.4rem; font-weight:700; color:#94a3b8; margin-top:12px;">No Dataset Loaded</div>
        <div style="color:#64748b; font-size:0.95rem; margin-top:8px;">
            Go to <strong>📂 Data Workspace</strong> in the sidebar and upload your company's CSV, Excel, or JSON file.
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("## 📈 Executive Business Intelligence Dashboard")
st.caption("Auto-adapts to your uploaded dataset — live KPIs, trends, and category breakdown.")

# ── Load Active Dataset ──────────────────────────────────────────────────────
active_name = dataset_manager.get_active_dataset_name()

if not active_name:
    _no_data_banner()
    st.stop()

df = dataset_manager.get_dataset_df(active_name)

if df.empty:
    _no_data_banner()
    st.stop()

roles = DynamicAnalyticsEngine.infer_column_roles(df)
kpis = DynamicAnalyticsEngine.get_dataset_kpis(df)

metric_col = roles["primary_metric"]
date_col   = roles["date_col"]
cat_col    = roles["primary_category"]

# ── KPI Cards ────────────────────────────────────────────────────────────────
st.markdown(f"#### 📊 Dataset: `{active_name}` · {kpis['total_records']:,} records · {kpis['time_span']}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    val = f"{kpis['metric_sum']:,.2f}" if metric_col else str(kpis["total_records"])
    render_metric_card(f"Total {kpis['metric_name']}", val, f"{kpis['growth_pct']:+0.1f}% trend", icon="💰")
with c2:
    render_metric_card("Total Records", f"{kpis['total_records']:,}", "All rows in dataset", icon="📦")
with c3:
    render_metric_card("Avg per Record", f"{kpis['metric_avg']:,.2f}", kpis["metric_name"], icon="📊")
with c4:
    render_metric_card("Numeric Columns", f"{len(roles['metric_cols'])}", "Measurable fields", icon="🔢")
with c5:
    render_metric_card("Category Columns", f"{len(roles['categorical_cols'])}", "Groupable fields", icon="🏷️")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────

# Row 1: Trend + Category Bar
row1_left, row1_right = st.columns([6, 4])

with row1_left:
    if date_col and metric_col:
        df_trend = df[[date_col, metric_col]].copy()
        df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors="coerce")
        df_trend = df_trend.dropna()
        # Group by month if enough points, else weekly
        freq = "ME" if len(df_trend) >= 60 else "W"
        df_grouped = df_trend.groupby(pd.Grouper(key=date_col, freq=freq))[metric_col].sum().reset_index()
        fig_trend = px.area(
            df_grouped,
            x=date_col,
            y=metric_col,
            title=f"<b>{metric_col} Over Time</b>",
            color_discrete_sequence=["#6366f1"],
        )
        fig_trend.update_traces(fill="tozeroy", line=dict(width=2))
        fig_trend.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("📅 No date column detected — add a date column in your file to see time trends.")
        if metric_col:
            fig_hist = px.histogram(df, x=metric_col, title=f"Distribution of {metric_col}", color_discrete_sequence=["#6366f1"])
            fig_hist.update_layout(**DARK_THEME_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True)

with row1_right:
    if cat_col and metric_col:
        df_cat = df.groupby(cat_col)[metric_col].sum().reset_index().sort_values(metric_col, ascending=False).head(10)
        fig_cat = px.bar(
            df_cat,
            x=metric_col,
            y=cat_col,
            orientation="h",
            title=f"<b>Top {cat_col} by {metric_col}</b>",
            color=metric_col,
            color_continuous_scale="purples",
        )
        fig_cat.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_cat, use_container_width=True)
    elif roles["categorical_cols"]:
        # Frequency of most unique categorical column
        best_cat = roles["categorical_cols"][0]
        df_freq = df[best_cat].value_counts().head(10).reset_index()
        df_freq.columns = [best_cat, "count"]
        fig_freq = px.pie(df_freq, names=best_cat, values="count", title=f"<b>{best_cat} Distribution</b>", hole=0.45)
        fig_freq.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("No categorical columns detected for grouping.")

# Row 2: Scatter + Numeric Correlation
row2_left, row2_right = st.columns(2)

with row2_left:
    num_cols = roles["metric_cols"]
    if len(num_cols) >= 2:
        df_corr = df[num_cols].corr()
        fig_corr = px.imshow(
            df_corr,
            title="<b>Numeric Columns Correlation Heatmap</b>",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need ≥ 2 numeric columns for correlation heatmap.")

with row2_right:
    if len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
        scatter_color = cat_col if cat_col and df[cat_col].nunique() <= 15 else None
        fig_scatter = px.scatter(
            df.sample(min(500, len(df))),
            x=x_col, y=y_col,
            color=scatter_color,
            title=f"<b>{x_col} vs {y_col}</b>",
            opacity=0.7,
        )
        fig_scatter.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Need ≥ 2 numeric columns for scatter plot.")

# ── Top Records Table ─────────────────────────────────────────────────────────
st.markdown(f"### 📋 Top Records by `{metric_col or 'Row Index'}`")
display_df = df.sort_values(metric_col, ascending=False).head(20) if metric_col else df.head(20)
st.dataframe(display_df, use_container_width=True)
