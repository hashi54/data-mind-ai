from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluator:
    """Computes comprehensive evaluation metrics for classification and regression models."""

    @staticmethod
    def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
                metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            except Exception:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0
        return metrics

    @staticmethod
    def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE safely
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = 0.0

        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 1.0

        return {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "mape": round(float(mape), 2),
            "r2_score": round(float(r2), 4),
        }
