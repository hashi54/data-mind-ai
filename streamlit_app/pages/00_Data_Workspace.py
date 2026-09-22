import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from app.data.dataset_manager import dataset_manager
from app.analytics.dynamic_analytics import DynamicAnalyticsEngine
from streamlit_app.components.cards import render_metric_card
from streamlit_app.components.charts import DARK_THEME_LAYOUT

st.set_page_config(page_title="Data Workspace | DataMind AI", page_icon="📂", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.upload-zone {
    border: 2px dashed rgba(99, 102, 241, 0.5);
    border-radius: 16px;
    padding: 36px 24px;
    text-align: center;
    background: rgba(99, 102, 241, 0.05);
    margin-bottom: 20px;
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: rgba(99, 102, 241, 0.9); }
.dataset-card {
    background: rgba(18, 24, 38, 0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.active-badge {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid #10b981;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
}
.sheet-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
            border: 1px solid rgba(99,102,241,0.3); border-radius: 16px;
            padding: 24px 32px; margin-bottom: 24px;">
    <div style="font-size:1.9rem; font-weight:800;
                background: linear-gradient(90deg,#60a5fa,#a855f7);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        📂 Data Workspace
    </div>
    <div style="color:#94a3b8; font-size:1rem; margin-top:6px;">
        Upload your company's datasets (CSV, Excel, JSON). All pages auto-adapt to your data — no coding needed.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Upload Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### ⬆️ Upload Your Company Data")

upload_col, tip_col = st.columns([3, 2])

with upload_col:
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="workspace_uploader",
    )
    st.markdown("**Supported formats:** CSV · Excel (.xlsx / .xls, multi-sheet) · JSON", unsafe_allow_html=False)
    st.markdown("**Max size:** 50 MB per file &nbsp;|&nbsp; **Multiple files:** ✅ Supported", unsafe_allow_html=False)
    st.markdown('</div>', unsafe_allow_html=True)

with tip_col:
    st.info(
        "**💡 Tips for best results:**\n\n"
        "- First row should be **column headers**\n"
        "- Include a **date/time column** for trend analysis\n"
        "- Include a **revenue/sales/amount** column for forecasting\n"
        "- Multi-sheet Excel files will create **one table per sheet**\n"
        "- No data is sent to any cloud — processed **100% locally**"
    )

# Upload Processing
if uploaded_files:
    newly_registered = []
    errors = []
    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        for file in uploaded_files:
            try:
                file_bytes = file.read()
                tables = dataset_manager.ingest_and_register_file(file_bytes, file.name)
                newly_registered.extend(tables)
            except Exception as e:
                errors.append(f"**{file.name}**: {e}")

    if newly_registered:
        st.success(f"✅ Successfully loaded **{len(newly_registered)} dataset table(s)** into workspace!")
        for t in newly_registered:
            sheet_label = f'<span class="sheet-badge">Sheet: {t["sheet_name"]}</span>' if t.get("sheet_name") else ""
            st.markdown(
                f'<div class="dataset-card">'
                f'📊 <strong>{t["table_name"]}</strong>{sheet_label} &nbsp;—&nbsp; '
                f'<span style="color:#94a3b8">{t["total_rows"]:,} rows × {t["total_columns"]} columns</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.rerun()

    if errors:
        for err in errors:
            st.error(f"❌ {err}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Workspace Datasets List
# ─────────────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([4, 1])
with h_left:
    st.markdown("### 🗄️ Workspace Datasets")
with h_right:
    all_datasets = dataset_manager.list_datasets()
    if all_datasets:
        if st.button("🗑️ Clear All", key="clear_all_workspace", help="Remove all uploaded datasets from workspace"):
            dataset_manager.clear_all_datasets()
            st.warning("All custom datasets removed from workspace.")
            st.rerun()

active_name = dataset_manager.get_active_dataset_name()

if not all_datasets:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #475569;">
        <div style="font-size: 3rem; margin-bottom: 12px;">📭</div>
        <div style="font-size:1.1rem; font-weight:600; color:#64748b;">No datasets uploaded yet</div>
        <div style="font-size:0.9rem; color:#475569; margin-top:6px;">
            Upload your first file above — then all dashboards will run on your real data.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for ds in all_datasets:
        tbl = ds["table_name"]
        is_active = (tbl == active_name)

        card_cols = st.columns([5, 2, 2, 1])
        with card_cols[0]:
            badge = '<span class="active-badge">● ACTIVE</span>' if is_active else ""
            st.markdown(
                f'<div style="padding:8px 0">'
                f'<strong style="font-size:1rem;">{tbl}</strong> {badge}<br>'
                f'<span style="color:#64748b; font-size:0.83rem;">'
                f'{ds["original_filename"]}'
                f'{" · Sheet: " + ds["sheet_name"] if ds.get("sheet_name") else ""}'
                f' · Uploaded {ds["uploaded_at"][:10]}'
                f'</span></div>',
                unsafe_allow_html=True
            )
        with card_cols[1]:
            st.markdown(
                f'<div style="padding:12px 0; color:#94a3b8; font-size:0.88rem;">'
                f'🗂 <strong>{ds["total_rows"]:,}</strong> rows<br>'
                f'📐 <strong>{ds["total_columns"]}</strong> columns'
                f'</div>',
                unsafe_allow_html=True
            )
        with card_cols[2]:
            if not is_active:
                if st.button(f"Set Active", key=f"activate_{tbl}"):
                    dataset_manager.set_active_dataset(tbl)
                    st.session_state["active_dataset"] = tbl
                    st.success(f"'{tbl}' is now the active workspace dataset.")
                    st.rerun()
            else:
                st.markdown('<div style="padding:12px 0; color:#10b981; font-weight:600;">✅ Active Dataset</div>', unsafe_allow_html=True)
        with card_cols[3]:
            if st.button("🗑️", key=f"delete_{tbl}", help=f"Delete '{tbl}' from workspace"):
                dataset_manager.delete_dataset(tbl)
                st.warning(f"Dataset '{tbl}' removed from workspace.")
                st.rerun()

        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Live Preview of Active Dataset
# ─────────────────────────────────────────────────────────────────────────────
if active_name:
    st.markdown(f"### 🔍 Data Preview — `{active_name}`")

    df_preview = dataset_manager.get_dataset_df(active_name, limit=200)

    if not df_preview.empty:
        roles = DynamicAnalyticsEngine.infer_column_roles(df_preview)

        # Column role chips
        info_cols = st.columns(4)
        with info_cols[0]:
            st.markdown(f"📅 **Date column:** `{roles['date_col'] or 'Not detected'}`")
        with info_cols[1]:
            st.markdown(f"📊 **Primary metric:** `{roles['primary_metric'] or 'Not detected'}`")
        with info_cols[2]:
            st.markdown(f"🏷️ **Category column:** `{roles['primary_category'] or 'Not detected'}`")
        with info_cols[3]:
            st.markdown(f"🔢 **Numeric columns:** `{len(roles['metric_cols'])}`")

        st.dataframe(df_preview.head(100), use_container_width=True)

        # Column selector for quick chart
        st.markdown("#### ⚡ Quick Chart")
        qc1, qc2, qc3 = st.columns(3)
        num_cols = [c for c in df_preview.columns if pd.api.types.is_numeric_dtype(df_preview[c])]
        cat_cols = [c for c in df_preview.columns if not pd.api.types.is_numeric_dtype(df_preview[c])]

        with qc1:
            x_axis = st.selectbox("X Axis / Category", options=cat_cols + list(df_preview.columns), key="qc_x")
        with qc2:
            y_axis = st.selectbox("Y Axis / Metric", options=num_cols if num_cols else list(df_preview.columns), key="qc_y")
        with qc3:
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box"], key="qc_type")

        try:
            if chart_type == "Bar":
                fig = px.bar(df_preview, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", color_discrete_sequence=["#6366f1"])
            elif chart_type == "Line":
                fig = px.line(df_preview.sort_values(x_axis), x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}", color_discrete_sequence=["#a855f7"])
            elif chart_type == "Scatter":
                fig = px.scatter(df_preview, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}", color_discrete_sequence=["#ec4899"])
            elif chart_type == "Histogram":
                fig = px.histogram(df_preview, x=y_axis, title=f"Distribution of {y_axis}", color_discrete_sequence=["#06b6d4"])
            else:
                fig = px.box(df_preview, x=x_axis if df_preview[x_axis].nunique() < 20 else None, y=y_axis, title=f"Box Plot: {y_axis}", color_discrete_sequence=["#f59e0b"])
            fig.update_layout(**DARK_THEME_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

        # Download cleaned dataset
        st.markdown("#### 💾 Download Cleaned Data")
        csv_bytes = df_preview.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_bytes,
            file_name=f"{active_name}_cleaned.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"No data found in table '{active_name}'.")
