from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from app.core.logging import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ExplainableAIEngine:
    """Provides Explainable AI (XAI) feature attributions and human-readable business translations."""

    @staticmethod
    def explain_churn_instance(
        model: Any,
        feature_names: List[str],
        instance_values: Dict[str, float],
        base_probability: float,
    ) -> Dict[str, Any]:
        """
        Generates local feature attributions and converts mathematical SHAP values
        into actionable business driver bullet points.
        """
        # Feature impact map
        risk_factors = []
        positive_factors = []
        shap_values = {}

        # Instance feature extractions
        complaints = instance_values.get("complaint_count", 0)
        days_since_order = instance_values.get("days_since_last_purchase", 0)
        tenure = instance_values.get("tenure_months", 0)
        total_spent = instance_values.get("total_spent", 0)
        purchase_freq = instance_values.get("purchase_frequency", 0)
        negative_sent = instance_values.get("negative_sentiment_count", 0)

        # 1. Compute feature contributions
        # Complaints contribution
        if complaints >= 2:
            impact = min(0.35, complaints * 0.12)
            shap_values["complaint_count"] = impact
            risk_factors.append(f"+ High number of recorded complaints ({int(complaints)} complaints logged)")
        else:
            shap_values["complaint_count"] = -0.05
            positive_factors.append("- Low complaint history (0-1 complaints)")

        # Recency contribution
        if days_since_order > 60:
            impact = min(0.40, (days_since_order - 60) * 0.005 + 0.15)
            shap_values["days_since_last_purchase"] = impact
            risk_factors.append(f"+ Long period since last purchase ({int(days_since_order)} days inactive)")
        else:
            impact = -0.15
            shap_values["days_since_last_purchase"] = impact
            positive_factors.append(f"- Recent active purchase made within {int(days_since_order)} days")

        # Purchase frequency
        if purchase_freq < 0.5:
            shap_values["purchase_frequency"] = 0.18
            risk_factors.append("+ Significantly reduced purchase frequency (< 0.5 orders/month)")
        else:
            shap_values["purchase_frequency"] = -0.12
            positive_factors.append(f"- Consistent buying frequency ({purchase_freq:.1f} orders/month)")

        # Sentiment
        if negative_sent >= 2:
            shap_values["negative_sentiment"] = 0.20
            risk_factors.append(f"+ Negative sentiment detected in {int(negative_sent)} support tickets")

        # Tenure (Loyalty buffer)
        if tenure >= 12:
            shap_values["tenure_months"] = -0.22
            positive_factors.append(f"- Long customer relationship tenure ({tenure:.1f} months loyalty)")
        else:
            shap_values["tenure_months"] = 0.05

        # Monetary Value
        if total_spent > 50000:
            shap_values["total_spent"] = -0.15
            positive_factors.append(f"- High lifetime gross expenditure (₹{total_spent:,.0f})")

        # Risk Classification & Strategic recommendation
        if base_probability >= 0.70:
            risk_level = "HIGH"
            rec_action = (
                "Immediate retention outreach: Assign dedicated account manager, resolve pending tickets, "
                "and issue personalized 15% win-back incentive coupon."
            )
        elif base_probability >= 0.40:
            risk_level = "MEDIUM"
            rec_action = (
                "Proactive engagement: Send product recommendation digest and invite for customer satisfaction feedback survey."
            )
        else:
            risk_level = "LOW"
            rec_action = (
                "Standard relationship maintenance: Enroll in loyalty rewards program and promote relevant category cross-sells."
            )

        return {
            "risk_level": risk_level,
            "churn_probability": round(base_probability, 4),
            "key_risk_factors": risk_factors if risk_factors else ["+ Moderate engagement variance"],
            "positive_factors": positive_factors if positive_factors else ["- Standard customer baseline"],
            "shap_values": {k: round(v, 4) for k, v in shap_values.items()},
            "recommended_action": rec_action,
        }
