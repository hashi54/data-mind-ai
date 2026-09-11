from datetime import datetime, date
from typing import Tuple
import pandas as pd
import numpy as np
from app.database.connection import execute_query
from app.core.logging import logger


class FeatureEngineeringPipeline:
    """Extracts analytical and machine learning features from star-schema database."""

    @staticmethod
    def build_customer_churn_features() -> pd.DataFrame:
        """
        Extracts customer behavioral, transaction, and interaction features for churn modeling.
        """
        query = """
        WITH cust_orders AS (
            SELECT 
                customer_id,
                MAX(order_date) AS last_order_date,
                COUNT(order_id) AS total_orders,
                SUM(total_amount) AS total_spent,
                AVG(total_amount) AS avg_order_val,
                SUM(CASE WHEN order_status = 'Refunded' THEN 1 ELSE 0 END) AS refund_count
            FROM orders
            GROUP BY customer_id
        ),
        cust_complaints AS (
            SELECT 
                customer_id,
                COUNT(interaction_id) AS total_interactions,
                SUM(CASE WHEN interaction_type IN ('Complaint', 'Return Request') THEN 1 ELSE 0 END) AS complaint_count,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_sentiment_count
            FROM customer_interactions
            GROUP BY customer_id
        )
        SELECT 
            c.customer_id,
            c.age,
            c.gender,
            c.state,
            c.customer_segment,
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
        """
        df = execute_query(query)
        
        # Calculate derived feature metrics
        today = pd.to_datetime(date.today())
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        df["last_order_date"] = pd.to_datetime(df["last_order_date"])

        df["tenure_months"] = ((today - df["signup_date"]).dt.days / 30.44).round(1)
        df["days_since_last_purchase"] = (today - df["last_order_date"]).dt.days
        df["purchase_frequency"] = (df["total_orders"] / np.maximum(df["tenure_months"], 1.0)).round(2)
        df["complaint_ratio"] = (df["complaint_count"] / np.maximum(df["total_interactions"], 1.0)).round(2)

        # Churn Target Definition (Inactive > 75 days or high complaints with no recent orders)
        df["churn"] = np.where(
            (df["days_since_last_purchase"] > 70) | 
            ((df["complaint_count"] >= 3) & (df["days_since_last_purchase"] > 40)) |
            (df["customer_segment"].isin(["At Risk", "Inactive"])),
            1,
            0
        )

        return df

    @staticmethod
    def build_sales_time_series_features() -> pd.DataFrame:
        """
        Extracts daily aggregated revenue time-series with calendar and lag features.
        """
        query = """
        SELECT 
            order_date,
            SUM(total_amount) AS revenue,
            COUNT(DISTINCT order_id) AS order_count,
            COUNT(DISTINCT customer_id) AS active_customers,
            SUM(CASE WHEN order_status = 'Refunded' THEN total_amount ELSE 0 END) AS refunded_amount
        FROM orders
        GROUP BY order_date
        ORDER BY order_date ASC
        """
        df = execute_query(query)
        df["order_date"] = pd.to_datetime(df["order_date"])
        df = df.set_index("order_date").asfreq("D").fillna(0).reset_index()

        # Calendar features
        df["day_of_week"] = df["order_date"].dt.dayofweek
        df["day_of_month"] = df["order_date"].dt.day
        df["month"] = df["order_date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Rolling window features
        df["revenue_rolling_7d_mean"] = df["revenue"].rolling(window=7, min_periods=1).mean()
        df["revenue_rolling_30d_mean"] = df["revenue"].rolling(window=30, min_periods=1).mean()
        df["revenue_rolling_7d_std"] = df["revenue"].rolling(window=7, min_periods=1).std().fillna(0)

        # Lag features
        df["revenue_lag_1d"] = df["revenue"].shift(1).fillna(df["revenue"].mean())
        df["revenue_lag_7d"] = df["revenue"].shift(7).fillna(df["revenue"].mean())
        df["revenue_lag_14d"] = df["revenue"].shift(14).fillna(df["revenue"].mean())

        return df
