import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from app.data.dataset_manager import dataset_manager
from app.analytics.dynamic_analytics import DynamicAnalyticsEngine
from app.analytics.eda import AutomatedEDAEngine
from app.data.ingestion import DataIngestionEngine
from streamlit_app.components.cards import render_metric_card
from streamlit_app.components.charts import DARK_THEME_LAYOUT

st.set_page_config(page_title="Automated EDA | DataMind AI", page_icon="🔬", layout="wide")

st.markdown("## 🔬 Automated Exploratory Data Analysis (EDA) Studio")
st.caption("Full-spectrum automated profiling with schema inference, data health scoring, and missingness analysis — runs on your dataset.")

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

if df.empty:
    st.warning("Active dataset is empty.")
    st.stop()

roles = DynamicAnalyticsEngine.infer_column_roles(df)

# ── Profile ───────────────────────────────────────────────────────────────────
st.markdown(f"#### 📋 Profiling: `{active_name}` — {len(df):,} rows × {len(df.columns)} columns")

# Data Health Score
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
completeness = max(0, 100 - (missing_cells / max(total_cells, 1)) * 100)
dup_rows = df.duplicated().sum()
dup_penalty = (dup_rows / max(len(df), 1)) * 20
health_score = round(min(100.0, completeness - dup_penalty), 1)

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Data Health Score", f"{health_score}/100", "Completeness + Uniqueness", icon="❤️")
with c2:
    render_metric_card("Missing Cells", f"{missing_cells:,}", f"{completeness:.1f}% complete", icon="⚠️" if missing_cells > 0 else "✅")
with c3:
    render_metric_card("Duplicate Rows", f"{dup_rows:,}", f"{dup_penalty:.1f}% penalty applied", icon="🔁" if dup_rows > 0 else "✅")
with c4:
    render_metric_card("Numeric / Cat Cols", f"{len(roles['metric_cols'])} / {len(roles['categorical_cols'])}", "Column type split", icon="🗂")

st.divider()

# ── Column Profile Table ──────────────────────────────────────────────────────
st.markdown("### 📊 Column-Level Profile")

profile_rows = []
for col in df.columns:
    col_missing = int(df[col].isnull().sum())
    col_pct_missing = round(col_missing / max(len(df), 1) * 100, 1)
    col_unique = int(df[col].nunique())
    dtype_str = str(df[col].dtype)

    if pd.api.types.is_numeric_dtype(df[col]):
        col_type = "Numeric"
        stat_info = f"Min: {df[col].min():.2f} | Max: {df[col].max():.2f} | Mean: {df[col].mean():.2f}"
    elif col in (roles.get("date_col") or []):
        col_type = "DateTime"
        stat_info = f"Range: {df[col].min()} → {df[col].max()}"
    else:
        col_type = "Categorical"
        top_val = df[col].value_counts().index[0] if col_unique > 0 else "—"
        stat_info = f"Top: '{top_val}' | {col_unique} unique values"

    profile_rows.append({
        "Column": col,
        "Type": col_type,
        "Dtype": dtype_str,
        "Missing": f"{col_missing} ({col_pct_missing}%)",
        "Unique Values": col_unique,
        "Summary": stat_info,
    })

st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

# ── Missingness Heatmap ────────────────────────────────────────────────────────
st.markdown("### 🔎 Missingness Overview")

miss_series = df.isnull().sum()
miss_pct    = (miss_series / len(df) * 100).round(2)
df_miss = pd.DataFrame({"Column": miss_series.index, "Missing Count": miss_series.values, "Missing %": miss_pct.values})
df_miss = df_miss[df_miss["Missing Count"] > 0].sort_values("Missing %", ascending=False)

if df_miss.empty:
    st.success("✅ No missing values found — your dataset is 100% complete!")
else:
    fig_miss = px.bar(
        df_miss, x="Column", y="Missing %",
        title="<b>Missing Values by Column</b>",
        color="Missing %", color_continuous_scale="Reds",
    )
    fig_miss.update_layout(**DARK_THEME_LAYOUT)
    st.plotly_chart(fig_miss, use_container_width=True)

st.divider()

# ── Numeric Distribution Explorer ─────────────────────────────────────────────
st.markdown("### 📈 Numeric Column Distributions")

num_cols = roles["metric_cols"]
if num_cols:
    selected_num = st.multiselect("Select columns to visualise", num_cols, default=num_cols[:min(4, len(num_cols))], key="eda_num_select")
    if selected_num:
        cols_per_row = 2
        for i in range(0, len(selected_num), cols_per_row):
            row_cols = selected_num[i:i + cols_per_row]
            grid = st.columns(cols_per_row)
            for j, col_name in enumerate(row_cols):
                with grid[j]:
                    fig_dist = px.histogram(
                        df, x=col_name, nbins=40,
                        title=f"<b>Distribution: {col_name}</b>",
                        color_discrete_sequence=["#6366f1"],
                        marginal="box",
                    )
                    fig_dist.update_layout(**DARK_THEME_LAYOUT)
                    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("No numeric columns detected for distribution plots.")

# ── Correlation Heatmap ───────────────────────────────────────────────────────
if len(num_cols) >= 2:
    st.markdown("### 🔗 Correlation Matrix")
    corr_cols = st.multiselect("Columns for correlation", num_cols, default=num_cols[:min(8, len(num_cols))], key="eda_corr_select")
    if len(corr_cols) >= 2:
        df_corr = df[corr_cols].corr()
        fig_corr = px.imshow(
            df_corr, title="<b>Pearson Correlation Heatmap</b>",
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig_corr.update_layout(**DARK_THEME_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)
