import pytest
import pandas as pd
import numpy as np
from app.analytics.eda import AutomatedEDAEngine
from app.analytics.statistics import StatisticalEngine
from app.analytics.anomaly import AnomalyDetectionEngine


def test_statistical_engine():
    """Tests numerical summary statistics calculation."""
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 1000.0])  # Contains outlier
    stats_dict = StatisticalEngine.compute_numerical_summary(s)
    assert stats_dict["count"] == 6
    assert stats_dict["mean"] > 0
    assert stats_dict["outliers_count"] >= 1


def test_automated_eda_profiler():
    """Tests deep profiling of a dataframe."""
    df_mock = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        "category": ["Electronics", "Apparel", "Electronics", "Home", "Apparel"],
        "is_active": [True, False, True, True, False],
    })
    profile = AutomatedEDAEngine.profile_dataframe(df_mock, dataset_name="Test Mock")
    assert profile["total_rows"] == 5
    assert profile["total_columns"] == 3
    assert profile["data_health_score"] >= 80.0
    assert "revenue" in profile["numerical_summary"]
    assert "category" in profile["categorical_summary"]


def test_anomaly_detection_engine():
    """Tests Isolation Forest & Z-Score anomaly scanner."""
    res = AnomalyDetectionEngine.detect_sales_anomalies()
    assert "total_anomalies_detected" in res
    assert isinstance(res["anomalies"], list)
