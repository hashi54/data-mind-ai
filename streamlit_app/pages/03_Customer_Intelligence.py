import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from app.database.seed_data import seed_database
from app.ml.inference.churn_predictor import ChurnInferenceEngine
from app.ml.inference.segmenter import CustomerSegmenterEngine
from app.ml.inference.clv_predictor import CLVPredictorEngine
from app.database.connection import execute_query

from streamlit_app.components.cards import render_metric_card, render_risk_badge, render_segment_badge
from streamlit_app.components.charts import plot_shap_waterfall

st.set_page_config(page_title="Customer Intelligence | DataMind AI", page_icon="👥", layout="wide")

# Ensure the database is seeded (needed for standalone Streamlit Cloud deployment)
try:
    seed_database()
except Exception:
    pass

st.markdown("## 👥 Customer Intelligence & Explainable Churn Suite")
st.caption("Machine learning churn risk scoring, SHAP explainability, RFM segmentation, and 12-month CLV projections")

# Fetch Customers List for Interactive Selector
try:
    cust_df = execute_query("SELECT customer_id, name, email, customer_segment, city, state FROM customers ORDER BY customer_id ASC;")
except Exception as e:
    st.error("⚠️ Customer data is still loading. Please wait 10 seconds and refresh the page.")
    st.info("💡 **Tip**: The database is being initialized for the first time on this server. This only takes a moment.")
    st.stop()

if cust_df is None or cust_df.empty:
    st.warning("🚧 No customer records found. The database may still be seeding. Please refresh in 10 seconds.")
    st.stop()

cust_options = {f"#{row['customer_id']} — {row['name']} ({row['customer_segment']}, {row['city']})": row['customer_id'] for _, row in cust_df.iterrows()}

selected_label = st.selectbox("Select Customer to Analyze:", options=list(cust_options.keys()), index=0)
selected_cid = cust_options[selected_label]

# Run Inferences
with st.spinner("Calculating ML models and SHAP feature attributions..."):
    churn_res = ChurnInferenceEngine.predict(customer_id=selected_cid)
    seg_res = CustomerSegmenterEngine.classify(customer_id=selected_cid)
    clv_res = CLVPredictorEngine.predict(customer_id=selected_cid)

# Customer Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Churn Probability", f"{churn_res['churn_probability']*100:.1f}%", f"{churn_res['risk_level']} Risk Level", icon="⚠️")
with c2:
    render_metric_card("12-Month CLV", f"₹{clv_res['predicted_12m_clv']:,.0f}", clv_res['customer_tier'], icon="💎")
with c3:
    render_metric_card("RFM Segment", seg_res['segment'], "Behavioral Cluster", icon="🏷️")
with c4:
    render_metric_card("Avg Order Value", f"₹{clv_res['avg_order_value']:,.0f}", "Historical Baseline", icon="🛒")

st.divider()

# Explainable AI & Factor Drivers
col_left, col_right = st.columns([5, 5])

with col_left:
    st.markdown("### 🧠 Explainable AI: Churn Risk Drivers (SHAP)")
    st.markdown(f"**Overall Assessment:** {render_risk_badge(churn_res['risk_level'])}", unsafe_allow_html=True)
    
    st.markdown("#### 🚨 Key Risk Factors (+ Increasing Churn)")
    for rf in churn_res["key_risk_factors"]:
        st.markdown(f"• <span style='color: #f87171; font-weight: 600;'>{rf}</span>", unsafe_allow_html=True)

    st.markdown("#### 🛡️ Positive Protective Factors (- Reducing Churn)")
    for pf in churn_res["positive_factors"]:
        st.markdown(f"• <span style='color: #34d399; font-weight: 600;'>{pf}</span>", unsafe_allow_html=True)

    st.markdown("#### 🎯 Prescribed Business Action")
    st.info(f"**Recommended Strategy:** {churn_res['recommended_action']}")

with col_right:
    # Render SHAP Waterfall Chart
    fig_shap = plot_shap_waterfall(churn_res["shap_values"])
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# Top At-Risk Customers Requiring Intervention
st.markdown("### 🚨 Urgent Attention: Top At-Risk Customers")
st.caption("Customers identified by the ML pipeline with high churn probability and elevated complaint frequency")

high_risk_query = """
SELECT 
    c.customer_id AS "ID",
    c.name AS "Customer Name",
    c.email AS "Email",
    c.customer_segment AS "Segment",
    c.city AS "City",
    COUNT(DISTINCT o.order_id) AS "Total Orders",
    COALESCE(SUM(o.total_amount), 0.0) AS "Total Spent (₹)"
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_segment IN ('At Risk', 'Inactive')
GROUP BY c.customer_id
LIMIT 8;
"""
df_at_risk = execute_query(high_risk_query)
st.dataframe(df_at_risk, use_container_width=True)
