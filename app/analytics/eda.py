from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from app.analytics.statistics import StatisticalEngine
from app.core.logging import logger


class AutomatedEDAEngine:
    """Generates automated exploratory data analysis, statistical profiles, and data health scores."""

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, dataset_name: str = "Uploaded Dataset") -> Dict[str, Any]:
        """Performs deep automated EDA on any tabular dataset."""
        logger.info(f"Generating automated EDA profile for {dataset_name} ({len(df)} rows, {len(df.columns)} cols)")

        total_rows = len(df)
        total_cols = len(df.columns)
        
        if total_rows == 0:
            return {
                "dataset_name": dataset_name,
                "total_rows": 0,
                "total_columns": 0,
                "data_health_score": 0.0,
                "error": "Dataset is empty.",
            }

        # 1. Column Types
        col_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_types[col] = "Numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_types[col] = "DateTime"
            elif pd.api.types.is_bool_dtype(df[col]):
                col_types[col] = "Boolean"
            else:
                col_types[col] = "Categorical / Text"

        # 2. Missing Values Analysis
        null_counts = df.isnull().sum()
        missing_report = {}
        for col in df.columns:
            missing_report[col] = {
                "missing_count": int(null_counts[col]),
                "missing_pct": round(float((null_counts[col] / total_rows) * 100), 2),
            }

        # 3. Duplicate Rows
        duplicate_count = int(df.duplicated().sum())

        # 4. Numerical Summaries
        num_summary = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            num_summary[col] = StatisticalEngine.compute_numerical_summary(df[col])

        # 5. Categorical Summaries
        cat_summary = {}
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for col in categorical_cols:
            cat_summary[col] = StatisticalEngine.compute_categorical_summary(df[col])

        # 6. Correlation Matrix (for numeric columns)
        corr_matrix = None
        if len(numeric_cols) > 1:
            try:
                corr_df = df[numeric_cols].corr(method="pearson").round(3)
                corr_matrix = corr_df.to_dict()
            except Exception as e:
                logger.warning(f"Failed to compute correlation matrix: {e}")

        # 7. Data Health Score Calculation (0 to 100)
        # Deduct for missing values, duplicates, and extreme outliers
        total_cells = total_rows * total_cols
        missing_cells_pct = (null_counts.sum() / total_cells) * 100
        duplicate_pct = (duplicate_count / total_rows) * 100

        health_score = 100.0 - (missing_cells_pct * 1.5) - (duplicate_pct * 2.0)
        health_score = max(10.0, min(100.0, round(health_score, 1)))

        return {
            "dataset_name": dataset_name,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "column_types": col_types,
            "missing_values_report": missing_report,
            "duplicate_rows_count": duplicate_count,
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
            "correlation_matrix": corr_matrix,
            "data_health_score": health_score,
        }
