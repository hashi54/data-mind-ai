from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from app.data.preprocessing import FeatureEngineeringPipeline
from app.ml.evaluation.metrics import ModelEvaluator
from app.ml.registry import registry
from app.core.logging import logger


def train_sales_forecasting_models() -> Dict[str, Any]:
    """
    Trains multiple sales forecasting models (Ridge baseline, Random Forest, Gradient Boosting)
    on daily revenue time series data.
    """
    logger.info("Extracting time series features for sales forecasting...")
    df = FeatureEngineeringPipeline.build_sales_time_series_features()

    feature_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "revenue_rolling_7d_mean",
        "revenue_rolling_30d_mean",
        "revenue_rolling_7d_std",
        "revenue_lag_1d",
        "revenue_lag_7d",
        "revenue_lag_14d",
    ]

    # Time-based train/test split (last 45 days as holdout)
    split_idx = len(df) - 45
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["revenue"]
    X_test = test_df[feature_cols]
    y_test = test_df["revenue"]

    candidate_models = {
        "sales_forecast_baseline_ridge": Ridge(alpha=1.0),
        "sales_forecast_random_forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
        "sales_forecast_gradient_boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.06, max_depth=4, random_state=42),
    }

    best_model_name = None
    best_rmse = float("inf")
    evaluation_summary = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(0, y_pred)  # Non-negative revenue

        metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
        evaluation_summary[name] = metrics
        logger.info(f"Model '{name}' -> MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}%, R2: {metrics['r2_score']}")

        payload = {
            "model": model,
            "feature_names": feature_cols,
            "last_history_row": df.iloc[-1].to_dict(),
        }
        registry.register_model(
            model_name=name,
            model_obj=payload,
            model_type="time_series_regression",
            features=feature_cols,
            metrics=metrics,
            parameters=model.get_params(),
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = name

    # Register default champion
    champion_payload = {
        "model": candidate_models[best_model_name],
        "feature_names": feature_cols,
        "last_history_row": df.iloc[-1].to_dict(),
    }
    registry.register_model(
        model_name="sales_forecast_champion",
        model_obj=champion_payload,
        model_type="time_series_regression",
        features=feature_cols,
        metrics=evaluation_summary[best_model_name],
        parameters={"champion_source": best_model_name},
    )

    logger.info(f"Champion Forecasting Model: {best_model_name} with RMSE of {best_rmse}")
    return evaluation_summary


if __name__ == "__main__":
    train_sales_forecasting_models()
