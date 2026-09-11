from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from app.data.preprocessing import FeatureEngineeringPipeline
from app.ml.evaluation.metrics import ModelEvaluator
from app.ml.registry import registry
from app.core.logging import logger


def train_clv_model() -> Dict[str, Any]:
    """
    Trains Gradient Boosting regression model to predict 12-month forward Customer Lifetime Value (CLV).
    """
    logger.info("Extracting customer behavioral data for CLV modeling...")
    df = FeatureEngineeringPipeline.build_customer_churn_features()

    feature_cols = [
        "age",
        "tenure_months",
        "days_since_last_purchase",
        "total_orders",
        "avg_order_val",
        "purchase_frequency",
        "complaint_count",
    ]

    X = df[feature_cols].fillna(0)
    # Synthetic target: expected 12-month value = (purchase_freq * avg_order_val * 12) adjusted by complaint penalty
    y = np.maximum(
        1000.0,
        (df["purchase_frequency"] * df["avg_order_val"] * 12) * (1 - df["complaint_ratio"] * 0.4)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)

    payload = {
        "model": model,
        "feature_names": feature_cols,
    }

    registry.register_model(
        model_name="customer_clv_champion",
        model_obj=payload,
        model_type="regression",
        features=feature_cols,
        metrics=metrics,
        parameters=model.get_params(),
    )
    logger.info(f"Registered CLV Model -> MAE: {metrics['mae']}, R2: {metrics['r2_score']}")
    return metrics


if __name__ == "__main__":
    train_clv_model()
