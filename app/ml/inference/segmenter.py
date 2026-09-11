from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from app.ml.registry import registry
from app.database.connection import execute_query
from app.core.logging import logger


class CustomerSegmenterEngine:
    """Classifies customers into RFM behavioral clusters and provides tactical strategies."""

    STRATEGIES = {
        "VIP": "Reward loyalty with exclusive early access, dedicated account rep, and luxury tier perks.",
        "Loyal": "Upsell premium product categories and offer VIP tier progression incentives.",
        "Potential Loyalist": "Engage with personalized recommendations, discount bundles, and reviews.",
        "At Risk": "Deploy urgent retention outreach, resolve recent friction points, and send win-back discounts.",
        "Inactive": "Run low-cost automated re-engagement email sequence with clearance flash promotions.",
    }

    DESCRIPTIONS = {
        "VIP": "Highest spenders with frequent repeat purchases and high loyalty score.",
        "Loyal": "Steady regular buyers with above-average order frequency and positive tenure.",
        "Potential Loyalist": "Recent buyers with promising frequency and healthy average basket value.",
        "At Risk": "Previously active customers whose last order recency has exceeded normal cycles.",
        "Inactive": "Dormant accounts with prolonged zero purchasing activity.",
    }

    @staticmethod
    def classify(customer_id: Optional[int] = None, rfm_values: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        model_data = registry.get_model("customer_segmentation_kmeans")
        if not model_data:
            raise ValueError("Customer segmentation model not found. Run training first.")

        kmeans = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        segment_map = model_data["segment_map"]

        if customer_id is not None:
            query = f"""
            SELECT 
                c.customer_id,
                (JULIANDAY('now') - JULIANDAY(COALESCE(MAX(o.order_date), c.signup_date))) AS days_since_last_purchase,
                COUNT(o.order_id) AS total_orders,
                COALESCE(SUM(o.total_amount), 0.0) AS total_spent,
                COALESCE(AVG(o.total_amount), 0.0) AS avg_order_val,
                ((JULIANDAY('now') - JULIANDAY(c.signup_date)) / 30.44) AS tenure_months
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            WHERE c.customer_id = {customer_id}
            GROUP BY c.customer_id
            """
            df = execute_query(query)
            if df.empty:
                raise ValueError(f"Customer {customer_id} not found.")
            row = df.iloc[0]
            rfm_dict = {
                "days_since_last_purchase": float(row["days_since_last_purchase"]),
                "total_orders": float(row["total_orders"]),
                "total_spent": float(row["total_spent"]),
                "avg_order_val": float(row["avg_order_val"]),
                "tenure_months": float(row["tenure_months"]),
            }
        elif rfm_values:
            rfm_dict = rfm_values
        else:
            raise ValueError("Must specify either customer_id or rfm_values.")

        vector = np.array([[rfm_dict[col] for col in feature_names]])
        vector_scaled = scaler.transform(vector)
        cluster_id = int(kmeans.predict(vector_scaled)[0])
        segment_name = segment_map.get(cluster_id, "Standard")

        return {
            "customer_id": customer_id,
            "segment": segment_name,
            "rfm_score": {k: round(v, 2) for k, v in rfm_dict.items()},
            "segment_description": CustomerSegmenterEngine.DESCRIPTIONS.get(segment_name, ""),
            "engagement_strategy": CustomerSegmenterEngine.STRATEGIES.get(segment_name, ""),
        }
