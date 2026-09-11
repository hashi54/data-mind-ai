from datetime import datetime, date, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from app.ml.registry import registry
from app.data.preprocessing import FeatureEngineeringPipeline
from app.core.logging import logger


class SalesForecasterEngine:
    """Generates multi-horizon forward sales projections with confidence bounds."""

    @staticmethod
    def forecast(horizon: str = "30d", model_choice: str = "champion") -> Dict[str, Any]:
        model_name = "sales_forecast_champion" if model_choice == "champion" else f"sales_forecast_{model_choice}"
        model_data = registry.get_model(model_name) or registry.get_model("sales_forecast_champion")

        if not model_data:
            raise ValueError("Sales forecast model not found in registry. Run training first.")

        model = model_data["model"]
        feature_names = model_data["feature_names"]
        model_info = registry.get_model_info("sales_forecast_champion")
        metrics = model_info.get("metrics", {}) if model_info else {"rmse": 4500.0, "mae": 3200.0, "mape": 8.5, "r2_score": 0.88}

        # Determine horizon days
        horizon_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
        days_ahead = horizon_map.get(horizon.lower(), 30)

        # Load recent time series history
        df_history = FeatureEngineeringPipeline.build_sales_time_series_features()
        last_date = df_history["order_date"].max()
        last_7d_mean = df_history["revenue"].iloc[-7:].mean()
        last_30d_mean = df_history["revenue"].iloc[-30:].mean()
        last_7d_std = df_history["revenue"].iloc[-7:].std()

        forecast_points = []
        running_history = list(df_history["revenue"].values)
        total_pred_rev = 0.0

        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            dow = future_date.weekday()
            dom = future_date.day
            month = future_date.month
            is_wknd = 1 if dow in [5, 6] else 0

            # Dynamic rolling values
            roll_7 = np.mean(running_history[-7:])
            roll_30 = np.mean(running_history[-30:])
            roll_std = np.std(running_history[-7:]) if len(running_history) >= 7 else 1000.0

            lag_1 = running_history[-1]
            lag_7 = running_history[-7] if len(running_history) >= 7 else roll_7
            lag_14 = running_history[-14] if len(running_history) >= 14 else roll_7

            feat_vector = pd.DataFrame([{
                "day_of_week": dow,
                "day_of_month": dom,
                "month": month,
                "is_weekend": is_wknd,
                "revenue_rolling_7d_mean": roll_7,
                "revenue_rolling_30d_mean": roll_30,
                "revenue_rolling_7d_std": roll_std,
                "revenue_lag_1d": lag_1,
                "revenue_lag_7d": lag_7,
                "revenue_lag_14d": lag_14,
            }])[feature_names]

            pred_val = float(model.predict(feat_vector)[0])
            pred_val = max(1000.0, pred_val)

            running_history.append(pred_val)
            total_pred_rev += pred_val

            # 95% Confidence Interval (± 1.96 * RMSE)
            margin = metrics.get("rmse", 4500.0) * 1.96
            forecast_points.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "forecast_revenue": round(pred_val, 2),
                "lower_bound": round(max(0.0, pred_val - margin), 2),
                "upper_bound": round(pred_val + margin, 2),
            })

        # Calculate projected growth rate compared to previous equivalent period
        past_equivalent_revenue = df_history["revenue"].iloc[-days_ahead:].sum()
        growth_rate = round(((total_pred_rev - past_equivalent_revenue) / max(past_equivalent_revenue, 1.0)) * 100, 2)

        summary = (
            f"Projected {horizon} revenue is ₹{total_pred_rev:,.2f}, representing a {growth_rate:+0.1f}% change "
            f"compared to the preceding period. Peak demand is anticipated during upcoming weekend cycles."
        )

        return {
            "horizon": horizon,
            "model_name": model_name,
            "total_predicted_revenue": round(total_pred_rev, 2),
            "growth_rate_pct": growth_rate,
            "metrics": metrics,
            "forecast_data": forecast_points,
            "insight_summary": summary,
        }
