import pytest
from app.ml.inference.churn_predictor import ChurnInferenceEngine
from app.ml.inference.forecaster import SalesForecasterEngine
from app.ml.inference.segmenter import CustomerSegmenterEngine
from app.ml.inference.clv_predictor import CLVPredictorEngine
from app.ml.evaluation.metrics import ModelEvaluator
from app.ml.registry import registry


def test_model_registry_loaded():
    """Verify champion models are registered."""
    models = registry.list_all_models()
    assert "churn_champion" in models or "churn_random_forest" in models
    assert "sales_forecast_champion" in models or "sales_forecast_random_forest" in models
    assert "customer_segmentation_kmeans" in models
    assert "customer_clv_champion" in models


def test_churn_inference_and_shap():
    """Tests churn prediction probability, risk level, and SHAP drivers."""
    res = ChurnInferenceEngine.predict(customer_id=1)
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_level"] in ["HIGH", "MEDIUM", "LOW"]
    assert len(res["key_risk_factors"]) > 0
    assert len(res["positive_factors"]) > 0
    assert "recommended_action" in res


def test_sales_forecaster():
    """Tests forward sales projection across horizons with confidence bounds."""
    res_30d = SalesForecasterEngine.forecast(horizon="30d")
    assert res_30d["total_predicted_revenue"] > 0
    assert len(res_30d["forecast_data"]) == 30
    assert "lower_bound" in res_30d["forecast_data"][0]
    assert "upper_bound" in res_30d["forecast_data"][0]


def test_segmentation_and_clv():
    """Tests RFM behavioral classification and CLV regression."""
    seg_res = CustomerSegmenterEngine.classify(customer_id=1)
    assert seg_res["segment"] in ["VIP", "Loyal", "Potential Loyalist", "At Risk", "Inactive", "Standard"]

    clv_res = CLVPredictorEngine.predict(customer_id=1)
    assert clv_res["predicted_12m_clv"] > 0
    assert "Tier" in clv_res["customer_tier"]


def test_evaluation_metrics():
    """Tests classification and regression metric calculators."""
    cls_metrics = ModelEvaluator.evaluate_classification([0, 1, 1, 0], [0, 1, 0, 0])
    assert "f1_score" in cls_metrics
    assert "precision" in cls_metrics

    reg_metrics = ModelEvaluator.evaluate_regression([100.0, 200.0, 300.0], [110.0, 190.0, 310.0])
    assert "mae" in reg_metrics
    assert "rmse" in reg_metrics
    assert "r2_score" in reg_metrics
