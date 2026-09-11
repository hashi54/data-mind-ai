import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st
import pandas as pd
from app.data.dataset_manager import dataset_manager
from app.analytics.dynamic_analytics import DynamicAnalyticsEngine
from streamlit_app.components.cards import render_metric_card
from app.ml.registry import registry
from app.core.config import settings

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind AI — Data Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366f1; }
    section[data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .feature-card {
        background: rgba(18,24,38,0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        transition: border-color 0.25s;
    }
    .feature-card:hover { border-color: rgba(99,102,241,0.5); }
    .step-circle {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        width: 40px; height: 40px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.1rem; color: white;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DataMind AI")
    st.caption("AI-Powered Data Intelligence Platform")
    st.divider()

    active_ds = dataset_manager.get_active_dataset_name()
    all_datasets = dataset_manager.list_datasets()

    st.markdown("#### 📂 Workspace")
    if active_ds:
        st.markdown(f"✅ **Active:** `{active_ds}`")
        ds_info = next((d for d in all_datasets if d["table_name"] == active_ds), {})
        if ds_info:
            st.markdown(f"   `{ds_info.get('total_rows', 0):,}` rows · `{ds_info.get('total_columns', 0)}` columns")
    else:
        st.warning("No dataset loaded yet")

    if len(all_datasets) > 1:
        st.markdown(f"📊 **{len(all_datasets)} datasets** in workspace")
    st.divider()

    st.markdown("#### ⚡ System")
    db_type = "PostgreSQL" if "postgres" in settings.DATABASE_URL.lower() else "SQLite"
    st.markdown(f"• **Database:** `{db_type}`")
    st.markdown(f"• **LLM:** `{settings.LLM_PROVIDER.upper()}`")
    models = registry.list_all_models()
    st.markdown(f"• **ML Models:** `{len(models)} trained`")
    st.divider()
    st.info("💡 Start by uploading your data in **📂 Data Workspace**")

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
            border: 1px solid rgba(99,102,241,0.3); border-radius: 16px;
            padding: 28px 36px; margin-bottom: 28px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.3);">
    <div style="font-size:2.3rem; font-weight:800;
                background: linear-gradient(90deg,#60a5fa,#a855f7,#ec4899);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px;">
        🧠 DataMind AI Platform
    </div>
    <div style="color:#94a3b8; font-size:1.05rem; max-width:800px; line-height:1.7;">
        Upload <strong>your company's data</strong> — CSV, Excel, or JSON — and instantly get AI-powered dashboards,
        forecasting, anomaly detection, automated EDA, and natural language business Q&amp;A.
        <strong>No coding. No cloud. 100% your data.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Onboarding / Live KPIs ───────────────────────────────────────────────────
active_name = dataset_manager.get_active_dataset_name()

if not active_name:
    # ─ Onboarding Flow ─
    st.markdown("## 🚀 Get Started in 3 Simple Steps")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="feature-card">
            <div class="step-circle">1</div>
            <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:8px;">📂 Upload Your Data</div>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                Go to <strong>Data Workspace</strong> in the sidebar.<br><br>
                Upload any <code>.csv</code>, <code>.xlsx</code>, <code>.xls</code> (multi-sheet), or <code>.json</code> file.
                You can upload multiple files at once.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="feature-card">
            <div class="step-circle">2</div>
            <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:8px;">🔍 Explore Instantly</div>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                Every dashboard auto-detects your date, revenue, and category columns.<br><br>
                Get instant KPI cards, trend charts, category breakdowns, and correlation heatmaps.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="feature-card">
            <div class="step-circle">3</div>
            <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:8px;">🤖 Ask AI Questions</div>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                Use the <strong>AI Analyst</strong> page to ask natural language questions about your data:<br><br>
                <em>"Why did sales drop last month?"</em><br>
                <em>"Which product is most profitable?"</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Upload shortcut inline
    st.markdown("### ⬆️ Or upload a file right here to begin:")
    quick_file = st.file_uploader(
        "Upload a file to get started",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="home_uploader",
    )
    if quick_file:
        with st.spinner("Loading your data..."):
            for f in quick_file:
                try:
                    tables = dataset_manager.ingest_and_register_file(f.read(), f.name)
                    for t in tables:
                        st.success(f"✅ Loaded `{t['table_name']}` — {t['total_rows']:,} rows × {t['total_columns']} columns")
                except Exception as e:
                    st.error(f"Error loading `{f.name}`: {e}")
        st.rerun()

else:
    # ─ Live KPIs from active dataset ─
    df = dataset_manager.get_dataset_df(active_name)
    kpis = DynamicAnalyticsEngine.get_dataset_kpis(df)
    roles = DynamicAnalyticsEngine.infer_column_roles(df)

    st.markdown(f"#### 📊 Active Dataset: `{active_name}` · {kpis['total_records']:,} records · {kpis['time_span']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_val = f"{kpis['metric_sum']:,.2f}" if roles["primary_metric"] else str(kpis["total_records"])
    with c1:
        render_metric_card(f"Total {kpis['metric_name']}", metric_val, f"{kpis['growth_pct']:+0.1f}% trend", icon="💰")
    with c2:
        render_metric_card("Total Records", f"{kpis['total_records']:,}", "All rows loaded", icon="📦")
    with c3:
        render_metric_card("Avg per Record", f"{kpis['metric_avg']:,.2f}", kpis["metric_name"], icon="📊")
    with c4:
        render_metric_card("Numeric Columns", f"{len(roles['metric_cols'])}", "Measurable fields", icon="🔢")
    with c5:
        render_metric_card("Category Columns", f"{len(roles['categorical_cols'])}", "Groupable dimensions", icon="🏷️")

    all_datasets = dataset_manager.list_datasets()
    if len(all_datasets) > 1:
        st.divider()
        st.markdown("#### 🗄️ All Workspace Datasets")
        for ds in all_datasets:
            is_active = ds["table_name"] == active_name
            badge = "● **ACTIVE**" if is_active else ""
            st.markdown(f"- `{ds['table_name']}` {badge} — {ds['total_rows']:,} rows · from `{ds['original_filename']}`")

# ── Platform Features Grid ────────────────────────────────────────────────────
st.divider()
st.markdown("### 🛠️ Platform Capabilities")

fa, fb, fc, fd = st.columns(4)
features = [
    ("📈", "Executive Dashboard", "KPI cards, trend charts, category breakdown, and correlation heatmaps auto-generated from your data."),
    ("🔮", "Sales Forecasting", "AI Random Forest model trains on your uploaded time series and predicts future values with confidence bands."),
    ("⚡", "Anomaly Detection", "Rolling Z-Score detects unusual spikes and drops in any metric column with root-cause hints."),
    ("🔬", "Automated EDA", "Full data health scoring, missingness analysis, distribution plots, and correlation heatmaps."),
]
for col, (icon, title, desc) in zip([fa, fb, fc, fd], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700; font-size:0.95rem; color:#f8fafc; margin-bottom:6px;">{title}</div>
            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

fe, ff, fg, fh = st.columns(4)
features2 = [
    ("🤖", "AI Business Analyst", "Ask natural language questions — the AI routes to SQL, ML, or RAG to generate precise answers."),
    ("👥", "Customer Intelligence", "Churn scoring, RFM segmentation, and CLV prediction with SHAP explainability."),
    ("📚", "Document Intelligence", "Semantic search across corporate policy documents using RAG with cosine similarity ranking."),
    ("📊", "Model Monitoring", "Track all trained ML model versions, hyperparameters, and benchmark performance metrics."),
]
for col, (icon, title, desc) in zip([fe, ff, fg, fh], features2):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700; font-size:0.95rem; color:#f8fafc; margin-bottom:6px;">{title}</div>
            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("👈 **Navigate using the sidebar — start with 📂 Data Workspace to load your data!**")
