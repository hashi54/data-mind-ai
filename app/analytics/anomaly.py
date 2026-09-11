from typing import Dict, Any, List
from datetime import datetime, date
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from app.database.connection import execute_query
from app.core.logging import logger


class AnomalyDetectionEngine:
    """Identifies revenue dips, order volume spikes, and unusual behavioral patterns."""

    @staticmethod
    def detect_sales_anomalies(contamination: float = 0.04) -> Dict[str, Any]:
        """
        Runs dual Isolation Forest and Rolling Z-Score anomaly detection on daily sales history.
        """
        query = """
        SELECT 
            order_date,
            SUM(total_amount) AS revenue,
            COUNT(DISTINCT order_id) AS order_count,
            SUM(CASE WHEN order_status = 'Refunded' THEN total_amount ELSE 0 END) AS refund_revenue
        FROM orders
        GROUP BY order_date
        ORDER BY order_date ASC
        """
        df = execute_query(query)
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["revenue"] = df["revenue"].astype(float)

        # 1. Rolling 14-day Baseline and Z-Score
        df["rolling_mean"] = df["revenue"].rolling(window=14, min_periods=3).mean()
        df["rolling_std"] = df["revenue"].rolling(window=14, min_periods=3).std().fillna(100.0)
        df["z_score"] = (df["revenue"] - df["rolling_mean"]) / np.maximum(df["rolling_std"], 10.0)

        # 2. Isolation Forest Model
        iso_features = df[["revenue", "order_count", "z_score"]].fillna(0)
        iso_model = IsolationForest(contamination=contamination, random_state=42)
        df["iso_anomaly"] = iso_model.fit_predict(iso_features)  # -1 for anomaly, 1 for normal

        anomalies_list = []
        for idx, row in df.iterrows():
            is_anomaly = (row["iso_anomaly"] == -1) or (abs(row["z_score"]) > 2.2)
            if is_anomaly:
                actual = float(row["revenue"])
                expected = float(row["rolling_mean"])
                diff_pct = round(((actual - expected) / max(expected, 1.0)) * 100, 1)

                # Determine Severity & Root-cause hint
                if diff_pct < -40.0:
                    sev = "Critical"
                    hint = f"Severe revenue drop of {abs(diff_pct)}% below 14-day baseline. High refund volume or supply delay."
                elif diff_pct > 60.0:
                    sev = "Warning"
                    hint = f"Significant revenue surge of +{diff_pct}%. Likely promotional campaign or seasonal demand spike."
                elif diff_pct < -20.0:
                    sev = "Moderate"
                    hint = f"Unusual dip of {abs(diff_pct)}% compared to typical baseline."
                else:
                    sev = "Moderate"
                    hint = "Multivariate outlier detected across order counts and ticket size."

                anomalies_list.append({
                    "date": row["order_date"].strftime("%Y-%m-%d"),
                    "metric": "Daily Revenue",
                    "actual_value": round(actual, 2),
                    "expected_value": round(expected, 2),
                    "deviation_pct": diff_pct,
                    "severity": sev,
                    "status": "Anomaly Detected",
                    "root_cause_hint": hint,
                })

        return {
            "total_anomalies_detected": len(anomalies_list),
            "anomalies": sorted(anomalies_list, key=lambda x: x["date"], reverse=True),
            "scan_timestamp": datetime.utcnow().isoformat(),
        }
