import pytest
import pandas as pd
import numpy as np
from app.data.validation import DataValidator
from app.data.cleaning import DataCleaner
from app.data.preprocessing import FeatureEngineeringPipeline


def test_data_validation():
    """Tests schema validation, missingness, and duplicate detection."""
    df_sample = pd.DataFrame({
        "customer_id": [1, 2, 2, 3],
        "name": ["Alice", "Bob", "Bob", None],
        "spend": [100.0, 250.0, 250.0, 50.0],
    })
    report = DataValidator.validate_schema(df_sample, expected_columns=["customer_id", "name"])
    assert report["is_valid"] is True
    assert report["summary"]["duplicate_rows"] == 1
    assert report["summary"]["missing_cells_total"] == 1


def test_data_cleaning():
    """Tests whitespace trimming, deduplication, and missing value imputation."""
    df_raw = pd.DataFrame({
        " Product Name ": [" Laptop ", "Phone", "Phone", " Tablet "],
        "Price": [50000.0, 20000.0, 20000.0, np.nan],
    })
    df_cleaned, summary = DataCleaner.clean(df_raw, drop_duplicates=True)
    assert len(df_cleaned) == 3
    assert "product_name" in df_cleaned.columns
    assert df_cleaned["price"].isnull().sum() == 0


def test_feature_engineering_pipelines():
    """Tests customer churn and time series feature generation from DB."""
    df_churn = FeatureEngineeringPipeline.build_customer_churn_features()
    assert not df_churn.empty
    assert "days_since_last_purchase" in df_churn.columns
    assert "tenure_months" in df_churn.columns
    assert "churn" in df_churn.columns

    df_ts = FeatureEngineeringPipeline.build_sales_time_series_features()
    assert not df_ts.empty
    assert "revenue_rolling_7d_mean" in df_ts.columns
    assert "revenue_lag_1d" in df_ts.columns
