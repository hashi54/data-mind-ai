import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import importlib
from app.database.seed_data import seed_database
from app.ml.registry import registry
from app.ml.inference.churn_predictor import ChurnInferenceEngine
from app.ml.inference.segmenter import CustomerSegmenterEngine
from app.ml.inference.clv_predictor import CLVPredictorEngine
from app.database.connection import execute_query
from app.core.logging import logger

from streamlit_app.components.cards import render_metric_card, render_risk_badge, render_segment_badge
from streamlit_app.components.charts import plot_shap_waterfall

st.set_page_config(page_title="Customer Intelligence | DataMind AI", page_icon="👥", layout="wide")


# ── Auto-Initialize DB Schema & Models ───────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initializing database schema and ML models...")
def _boot_customer_intelligence():
    """Ensures DB schema exists and required ML models are ready."""
    try:
        init_db()
    except Exception as e:
        logger.error(f"DB schema init error: {e}")

    models_needed = [
        ("churn_champion",               "train_customer_churn_models",   "app.ml.training.train_churn"),
        ("customer_clv_champion",        "train_clv_model",               "app.ml.training.train_clv"),
        ("customer_segmentation_kmeans", "train_customer_segmentation",   "app.ml.training.train_segmentation"),
    ]
    for model_key, func_name, module_path in models_needed:
        if registry.get_model(model_key) is None:
            try:
                mod = importlib.import_module(module_path)
                getattr(mod, func_name)()
            except Exception as e:
                logger.error(f"Auto-train failed for '{model_key}': {e}")
    return True

_boot_customer_intelligence()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 👥 Customer Intelligence & Explainable Churn Suite")
st.caption("Machine learning churn risk scoring, SHAP explainability, RFM segmentation, and 12-month CLV projections")

# ── Fetch Customer List ───────────────────────────────────────────────────────
try:
    cust_df = execute_query(
        "SELECT customer_id, name, email, customer_segment, city, state "
        "FROM customers ORDER BY customer_id ASC;"
    )
except Exception:
    cust_df = pd.DataFrame()

if cust_df is None or cust_df.empty:
    st.markdown("""
    <div style="text-align:center; padding:70px 20px;">
        <div style="font-size:3.5rem;">👥</div>
        <div style="font-size:1.4rem; font-weight:700; color:#94a3b8; margin-top:12px;">No Customer Dataset Loaded</div>
        <div style="color:#64748b; font-size:0.95rem; margin-top:8px; max-width:550px; margin-left:auto; margin-right:auto;">
            Go to <strong>📂 Data Workspace</strong> in the sidebar and upload your company's customer CSV or Excel file. 
            Once uploaded, select your active customer dataset to calculate ML churn risk, SHAP drivers, and CLV projections.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Optional button for users who DO want to load sample demo data
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='text-align:center; color:#94a3b8; font-size:0.9rem;'>Want to test the feature with sample data first?</div>", unsafe_allow_html=True)
        if st.button("🚀 Load Sample Enterprise Customer Dataset", use_container_width=True):
            with st.spinner("Generating sample customer enterprise data..."):
                seed_database()
                st.success("Sample customer dataset loaded successfully!")
                st.rerun()
    st.stop()

cust_options = {
    f"#{row['customer_id']} — {row['name']} ({row['customer_segment']}, {row['city']})": row["customer_id"]
    for _, row in cust_df.iterrows()
}

selected_label = st.selectbox("Select Customer to Analyze:", options=list(cust_options.keys()), index=0)
selected_cid = cust_options[selected_label]

# ── Run ML Inference ──────────────────────────────────────────────────────────
with st.spinner("Calculating ML models and SHAP feature attributions..."):
    churn_res, seg_res, clv_res = None, None, None
    errors = []

    try:
        churn_res = ChurnInferenceEngine.predict(customer_id=selected_cid)
    except Exception as e:
        errors.append(f"Churn model: {e}")

    try:
        seg_res = CustomerSegmenterEngine.classify(customer_id=selected_cid)
    except Exception as e:
        errors.append(f"Segmentation model: {e}")

    try:
        clv_res = CLVPredictorEngine.predict(customer_id=selected_cid)
    except Exception as e:
        errors.append(f"CLV model: {e}")

if errors:
    for err in errors:
        st.warning(f"⚠️ {err}")
    st.info("🔄 Some models could not run. Try refreshing — models may still be training.")

# ── Metric Cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    if churn_res:
        risk_color = "inverse" if churn_res["risk_level"] in ("HIGH", "MEDIUM") else "normal"
        render_metric_card(
            "Churn Probability",
            f"{churn_res['churn_probability']*100:.1f}%",
            f"{churn_res['risk_level']} Risk Level",
            delta_color=risk_color,
            icon="⚠️"
        )
    else:
        render_metric_card("Churn Probability", "N/A", "Model loading...", icon="⚠️")

with c2:
    if clv_res:
        render_metric_card(
            "12-Month CLV",
            f"₹{clv_res['predicted_12m_clv']:,.0f}",
            f"{clv_res['customer_tier']} Tier",
            delta_color="normal",
            icon="💎"
        )
    else:
        render_metric_card("12-Month CLV", "N/A", "Model loading...", icon="💎")

with c3:
    if seg_res:
        render_metric_card(
            "RFM Behavioral Segment",
            seg_res["segment"],
            "ML Cluster (Recency, Frequency, Monetary)",
            delta_color="normal",
            icon="🏷️"
        )
    else:
        render_metric_card("RFM Behavioral Segment", "N/A", "Model loading...", icon="🏷️")

with c4:
    if clv_res:
        render_metric_card(
            "Avg Order Value",
            f"₹{clv_res['avg_order_value']:,.0f}",
            "Historical Average per Order",
            delta_color="normal",
            icon="🛒"
        )
    else:
        render_metric_card("Avg Order Value", "N/A", "Model loading...", icon="🛒")

st.divider()

# ── SHAP Explainability Panel ─────────────────────────────────────────────────
if churn_res:
    col_left, col_right = st.columns([5, 5])

    with col_left:
        st.markdown("### 🧠 Explainable AI: Churn Risk Drivers (SHAP)")
        st.markdown(
            f"**Overall Assessment:** {render_risk_badge(churn_res['risk_level'])}",
            unsafe_allow_html=True
        )

        risk_factors = churn_res.get("key_risk_factors", [])
        pos_factors  = churn_res.get("positive_factors", [])

        if risk_factors:
            st.markdown("#### 🚨 Key Risk Factors (→ Increasing Churn)")
            for rf in risk_factors:
                st.markdown(f"• <span style='color:#f87171;font-weight:600'>{rf}</span>", unsafe_allow_html=True)
        else:
            st.success("✅ No significant churn risk factors identified.")

        if pos_factors:
            st.markdown("#### 🛡️ Protective Factors (→ Reducing Churn)")
            for pf in pos_factors:
                st.markdown(f"• <span style='color:#34d399;font-weight:600'>{pf}</span>", unsafe_allow_html=True)

        st.markdown("#### 🎯 Prescribed Business Action")
        st.info(f"**Recommended Strategy:** {churn_res.get('recommended_action', 'N/A')}")

    with col_right:
        try:
            fig_shap = plot_shap_waterfall(churn_res["shap_values"])
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            st.info("SHAP chart unavailable for this customer.")

    # Engagement strategy from segmentation
    if seg_res and seg_res.get("engagement_strategy"):
        st.markdown("#### 🤝 Engagement Strategy")
        st.success(f"**For {seg_res['segment']} customers:** {seg_res['engagement_strategy']}")

st.divider()

# ── Top At-Risk Customers Table ───────────────────────────────────────────────
st.markdown("### 🚨 Urgent Attention: Top At-Risk Customers")
st.caption(
    "Customers in 'At Risk' or 'Inactive' segments with highest complaint frequency. "
    "These are sourced from the built-in customer database."
)

high_risk_query = """
SELECT
    c.customer_id                            AS "ID",
    c.name                                   AS "Customer Name",
    c.email                                  AS "Email",
    c.customer_segment                       AS "Segment",
    c.city                                   AS "City",
    c.state                                  AS "State",
    COUNT(DISTINCT o.order_id)               AS "Total Orders",
    COALESCE(ROUND(SUM(o.total_amount),2), 0) AS "Total Spent (₹)",
    COUNT(DISTINCT ci.interaction_id)        AS "Complaints"
FROM customers c
LEFT JOIN orders o   ON c.customer_id = o.customer_id
LEFT JOIN customer_interactions ci
       ON c.customer_id = ci.customer_id AND ci.interaction_type = 'Complaint'
WHERE c.customer_segment IN ('At Risk', 'Inactive', 'Churned')
GROUP BY c.customer_id, c.name, c.email, c.customer_segment, c.city, c.state
ORDER BY "Complaints" DESC, "Total Spent (₹)" DESC
LIMIT 10;
"""

try:
    df_at_risk = execute_query(high_risk_query)
    if df_at_risk is not None and not df_at_risk.empty:
        st.dataframe(df_at_risk, use_container_width=True)
    else:
        # Fallback: show ANY customers with high complaint counts
        fallback_query = """
        SELECT
            c.customer_id   AS "ID",
            c.name          AS "Customer Name",
            c.customer_segment AS "Segment",
            c.city          AS "City",
            COUNT(DISTINCT ci.interaction_id) AS "Complaints",
            COALESCE(ROUND(SUM(o.total_amount),2), 0) AS "Total Spent (₹)"
        FROM customers c
        LEFT JOIN customer_interactions ci
               ON c.customer_id = ci.customer_id AND ci.interaction_type = 'Complaint'
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id
        ORDER BY "Complaints" DESC
        LIMIT 10;
        """
        df_fallback = execute_query(fallback_query)
        if df_fallback is not None and not df_fallback.empty:
            st.info("ℹ️ No 'At Risk/Inactive' segment customers found. Showing customers with highest complaint counts instead:")
            st.dataframe(df_fallback, use_container_width=True)
        else:
            st.info("ℹ️ No at-risk customer data available yet.")
except Exception as e:
    st.warning(f"⚠️ Could not load at-risk customers: {e}")
