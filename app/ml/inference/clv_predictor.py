from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from app.ml.registry import registry
from app.database.connection import execute_query
from app.core.logging import logger


class CLVPredictorEngine:
    """Predicts forward 12-month Customer Lifetime Value (CLV) and tier assignment."""

    @staticmethod
    def predict(customer_id: Optional[int] = None, custom_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        model_data = registry.get_model("customer_clv_champion")
        if not model_data:
            logger.info("CLV model not found in registry. Auto-triggering training...")
            try:
                from app.ml.training.train_clv import train_clv_model
                train_clv_model()
                model_data = registry.get_model("customer_clv_champion")
            except Exception as e:
                logger.error(f"Auto-training CLV model failed: {e}")

        if not model_data:
            raise ValueError("CLV model not found in registry and could not be auto-trained.")

        model = model_data["model"]
        feature_names = model_data["feature_names"]

        if customer_id is not None:
            query = f"""
            WITH cust_orders AS (
                SELECT 
                    customer_id,
                    COUNT(order_id) AS total_orders,
                    AVG(total_amount) AS avg_order_val,
                    MAX(order_date) AS last_order_date
                FROM orders
                WHERE customer_id = {customer_id}
                GROUP BY customer_id
            ),
            cust_comp AS (
                SELECT 
                    customer_id,
                    COUNT(interaction_id) AS complaint_count
                FROM customer_interactions
                WHERE customer_id = {customer_id} AND interaction_type = 'Complaint'
                GROUP BY customer_id
            )
            SELECT 
                c.customer_id,
                c.age,
                c.signup_date,
                COALESCE(co.total_orders, 0) AS total_orders,
                COALESCE(co.avg_order_val, 0.0) AS avg_order_val,
                COALESCE(co.last_order_date, c.signup_date) AS last_order_date,
                COALESCE(cc.complaint_count, 0) AS complaint_count
            FROM customers c
            LEFT JOIN cust_orders co ON c.customer_id = co.customer_id
            LEFT JOIN cust_comp cc ON c.customer_id = cc.customer_id
            WHERE c.customer_id = {customer_id}
            """
            df = execute_query(query)
            if df.empty:
                raise ValueError(f"Customer {customer_id} not found.")
            row = df.iloc[0]
            today = pd.to_datetime("today")
            signup_dt = pd.to_datetime(row["signup_date"])
            last_order_dt = pd.to_datetime(row["last_order_date"])

            tenure = round((today - signup_dt).days / 30.44, 1)
            recency = (today - last_order_dt).days
            freq = round(row["total_orders"] / max(tenure, 1.0), 2)

            feat_dict = {
                "age": float(row["age"] or 35),
                "tenure_months": float(tenure),
                "days_since_last_purchase": float(recency),
                "total_orders": float(row["total_orders"]),
                "avg_order_val": float(row["avg_order_val"]),
                "purchase_frequency": float(freq),
                "complaint_count": float(row["complaint_count"]),
            }
        elif custom_features:
            feat_dict = custom_features
        else:
            raise ValueError("Must provide customer_id or custom_features.")

        vector = pd.DataFrame([feat_dict])[feature_names]
        predicted_clv = float(model.predict(vector)[0])
        predicted_clv = max(500.0, round(predicted_clv, 2))

        # Assign Customer Tier
        if predicted_clv >= 100000:
            tier = "Diamond Tier"
            strategy = "Executive account coverage, white-glove onboarding for new product rollouts."
        elif predicted_clv >= 50000:
            tier = "Platinum Tier"
            strategy = "Priority customer support queue, exclusive early seasonal discounts."
        elif predicted_clv >= 20000:
            tier = "Gold Tier"
            strategy = "Automated cross-sell campaigns, loyalty point incentives."
        else:
            tier = "Silver Tier"
            strategy = "Re-engagement email journeys and low-friction catalog promotions."

        return {
            "customer_id": customer_id,
            "predicted_12m_clv": predicted_clv,
            "customer_tier": tier,
            "avg_order_value": round(feat_dict.get("avg_order_val", 0.0), 2),
            "recommended_strategy": strategy,
        }
