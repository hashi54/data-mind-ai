from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from app.data.preprocessing import FeatureEngineeringPipeline
from app.ml.evaluation.metrics import ModelEvaluator
from app.ml.registry import registry
from app.core.logging import logger


def train_customer_churn_models() -> Dict[str, Any]:
    """
    Trains multiple Churn Prediction algorithms (Logistic Regression, Random Forest, Gradient Boosting)
    and saves the champion model to the registry.
    """
    logger.info("Extracting customer features for churn training...")
    df = FeatureEngineeringPipeline.build_customer_churn_features()

    feature_cols = [
        "age",
        "tenure_months",
        "days_since_last_purchase",
        "total_orders",
        "total_spent",
        "avg_order_val",
        "refund_count",
        "complaint_count",
        "negative_sentiment_count",
        "purchase_frequency",
        "complaint_ratio",
    ]

    X = df[feature_cols].fillna(0)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        "churn_logistic_regression": LogisticRegression(random_state=42, max_iter=500),
        "churn_random_forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "churn_gradient_boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
    }

    best_model_name = None
    best_f1 = -1.0
    evaluation_summary = {}

    for name, model in candidate_models.items():
        is_linear = "logistic" in name
        X_tr = X_train_scaled if is_linear else X_train
        X_te = X_test_scaled if is_linear else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = ModelEvaluator.evaluate_classification(y_test, y_pred, y_prob)
        evaluation_summary[name] = metrics
        logger.info(f"Model '{name}' -> F1: {metrics['f1_score']}, ROC-AUC: {metrics.get('roc_auc')}, PR-AUC: {metrics.get('pr_auc')}")

        # Register each model
        model_payload = {
            "model": model,
            "scaler": scaler if is_linear else None,
            "feature_names": feature_cols,
        }
        registry.register_model(
            model_name=name,
            model_obj=model_payload,
            model_type="binary_classification",
            features=feature_cols,
            metrics=metrics,
            parameters=model.get_params(),
        )

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name

    # Also register the default champion as 'churn_champion'
    champion_payload = {
        "model": candidate_models[best_model_name],
        "scaler": scaler if "logistic" in best_model_name else None,
        "feature_names": feature_cols,
    }
    registry.register_model(
        model_name="churn_champion",
        model_obj=champion_payload,
        model_type="binary_classification",
        features=feature_cols,
        metrics=evaluation_summary[best_model_name],
        parameters={"champion_source": best_model_name},
    )

    logger.info(f"Champion Churn Model: {best_model_name} with F1-Score of {best_f1}")
    return evaluation_summary


if __name__ == "__main__":
    train_customer_churn_models()
