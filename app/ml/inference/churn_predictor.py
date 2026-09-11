from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from app.ml.registry import registry
from app.ml.explainability import ExplainableAIEngine
from app.database.connection import execute_query
from app.core.logging import logger


class ChurnInferenceEngine:
    """Performs real-time churn scoring and Explainable AI factor breakdown."""

    @staticmethod
    def predict(customer_id: Optional[int] = None, custom_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        champion_data = registry.get_model("churn_champion")
        if not champion_data:
            raise ValueError("Churn model not registered. Please run model training first.")

        model = champion_data["model"]
        scaler = champion_data.get("scaler")
        feature_names = champion_data["feature_names"]

        feature_dict = {}

        if customer_id is not None:
            # Fetch features directly from database
            query = f"""
            WITH cust_orders AS (
                SELECT 
                    customer_id,
                    MAX(order_date) AS last_order_date,
                    COUNT(order_id) AS total_orders,
                    SUM(total_amount) AS total_spent,
                    AVG(total_amount) AS avg_order_val,
                    SUM(CASE WHEN order_status = 'Refunded' THEN 1 ELSE 0 END) AS refund_count
                FROM orders
                WHERE customer_id = {customer_id}
                GROUP BY customer_id
            ),
            cust_complaints AS (
                SELECT 
                    customer_id,
                    COUNT(interaction_id) AS total_interactions,
                    SUM(CASE WHEN interaction_type IN ('Complaint', 'Return Request') THEN 1 ELSE 0 END) AS complaint_count,
                    SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_sentiment_count
                FROM customer_interactions
                WHERE customer_id = {customer_id}
                GROUP BY customer_id
            )
            SELECT 
                c.customer_id,
                c.age,
                c.signup_date,
                COALESCE(co.last_order_date, c.signup_date) AS last_order_date,
                COALESCE(co.total_orders, 0) AS total_orders,
                COALESCE(co.total_spent, 0.0) AS total_spent,
                COALESCE(co.avg_order_val, 0.0) AS avg_order_val,
                COALESCE(co.refund_count, 0) AS refund_count,
                COALESCE(cc.total_interactions, 0) AS total_interactions,
                COALESCE(cc.complaint_count, 0) AS complaint_count,
                COALESCE(cc.negative_sentiment_count, 0) AS negative_sentiment_count
            FROM customers c
            LEFT JOIN cust_orders co ON c.customer_id = co.customer_id
            LEFT JOIN cust_complaints cc ON c.customer_id = cc.customer_id
            WHERE c.customer_id = {customer_id}
            """
            df = execute_query(query)
            if df.empty:
                raise ValueError(f"Customer with ID {customer_id} not found.")

            row = df.iloc[0]
            today = pd.to_datetime("today")
            signup_dt = pd.to_datetime(row["signup_date"])
            last_order_dt = pd.to_datetime(row["last_order_date"])

            tenure = round((today - signup_dt).days / 30.44, 1)
            recency = (today - last_order_dt).days
            freq = round(row["total_orders"] / max(tenure, 1.0), 2)
            comp_ratio = round(row["complaint_count"] / max(row["total_interactions"], 1), 2)

            feature_dict = {
                "age": float(row["age"] or 35),
                "tenure_months": float(tenure),
                "days_since_last_purchase": float(recency),
                "total_orders": float(row["total_orders"]),
                "total_spent": float(row["total_spent"]),
                "avg_order_val": float(row["avg_order_val"]),
                "refund_count": float(row["refund_count"]),
                "complaint_count": float(row["complaint_count"]),
                "negative_sentiment_count": float(row["negative_sentiment_count"]),
                "purchase_frequency": float(freq),
                "complaint_ratio": float(comp_ratio),
            }
        elif custom_features:
            feature_dict = {
                "age": float(custom_features.get("age", 35)),
                "tenure_months": float(custom_features.get("tenure_months", 12)),
                "days_since_last_purchase": float(custom_features.get("days_since_last_purchase", 30)),
                "total_orders": float(custom_features.get("total_orders", 5)),
                "total_spent": float(custom_features.get("total_spent", 15000)),
                "avg_order_val": float(custom_features.get("avg_order_val", 3000)),
                "refund_count": float(custom_features.get("refund_count", 0)),
                "complaint_count": float(custom_features.get("complaint_count", 0)),
                "negative_sentiment_count": float(custom_features.get("negative_sentiment_count", 0)),
                "purchase_frequency": float(custom_features.get("purchase_frequency", 0.5)),
                "complaint_ratio": float(custom_features.get("complaint_ratio", 0.0)),
            }
        else:
            raise ValueError("Must provide either customer_id or custom_features.")

        # Build feature vector as DataFrame to preserve feature names
        vector_df = pd.DataFrame([feature_dict])[feature_names]
        if scaler:
            scaled_vals = scaler.transform(vector_df)
            vector_df = pd.DataFrame(scaled_vals, columns=feature_names)

        prob = float(model.predict_proba(vector_df)[0, 1])

        # Generate Explainable AI outputs
        xai_result = ExplainableAIEngine.explain_churn_instance(
            model=model,
            feature_names=feature_names,
            instance_values=feature_dict,
            base_probability=prob,
        )

        return {
            "customer_id": customer_id,
            "churn_probability": xai_result["churn_probability"],
            "risk_level": xai_result["risk_level"],
            "key_risk_factors": xai_result["key_risk_factors"],
            "positive_factors": xai_result["positive_factors"],
            "shap_values": xai_result["shap_values"],
            "recommended_action": xai_result["recommended_action"],
        }
