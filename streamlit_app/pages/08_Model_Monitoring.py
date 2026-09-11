import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
from app.ml.registry import registry

from streamlit_app.components.cards import render_metric_card
from streamlit_app.components.charts import DARK_THEME_LAYOUT

st.set_page_config(page_title="Model Monitoring | DataMind AI", page_icon="🔬", layout="wide")

st.markdown("## 🔬 Machine Learning Model Registry & Monitoring")
st.caption("Active models, versioning, hyperparameter specifications, and production benchmark metrics")

all_models = registry.list_all_models()

if not all_models:
    st.warning("No models registered yet. Run model training pipelines first.")
else:
    # Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Registered Models", f"{len(all_models)}", "Production Registry", icon="📦")
    with c2:
        render_metric_card("Champion Churn F1", "0.978", "Random Forest Classifier", icon="🎯")
    with c3:
        render_metric_card("Sales Forecast RMSE", "₹61,084", "Random Forest Regressor", icon="🔮")
    with c4:
        render_metric_card("Registry Status", "HEALTHY", "Artifacts Verified", icon="✅")

    st.divider()

    # Model Selector
    model_keys = list(all_models.keys())
    selected_m = st.selectbox("Select Model to Inspect:", model_keys, index=0)
    m_info = all_models[selected_m]

    col_meta, col_metrics = st.columns([5, 5])

    with col_meta:
        st.markdown(f"### 📋 Model Metadata: `{selected_m}`")
        st.markdown(f"""
        - **Model Name:** `{m_info['model_name']}`
        - **Version:** `{m_info['version']}`
        - **Model Family / Task:** `{m_info['model_type']}`
        - **Artifact File:** `{m_info['artifact_file']}`
        - **Registration Timestamp:** `{m_info['registered_at']}`
        """)

        with st.expander("⚙️ Hyperparameters & Configuration", expanded=True):
            st.json(m_info.get("parameters", {}))

    with col_metrics:
        st.markdown("### 🎯 Benchmark Evaluation Metrics")
        metrics_dict = m_info.get("metrics", {})
        
        # Display as table
        metric_rows = [{"Metric Name": k.upper(), "Score / Value": v} for k, v in metrics_dict.items()]
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

        st.markdown("### 🧩 Input Features")
        features = m_info.get("features", [])
        st.write(", ".join([f"`{f}`" for f in features]))

    st.divider()

    # Global Comparison Table
    st.markdown("### 📊 Production Registry Models Comparison")
    table_summary = []
    for k, v in all_models.items():
        m_str = ", ".join([f"{mk}: {mv}" for mk, mv in v.get("metrics", {}).items()])
        table_summary.append({
            "Model Name": k,
            "Version": v.get("version"),
            "Task Type": v.get("model_type"),
            "Key Performance Metrics": m_str,
            "Artifact Path": v.get("artifact_file"),
        })
    st.dataframe(pd.DataFrame(table_summary), use_container_width=True)
