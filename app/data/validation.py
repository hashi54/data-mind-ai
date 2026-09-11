from typing import Dict, Any, List
import pandas as pd
import numpy as np
from app.core.logging import logger


class DataValidator:
    """Validates structural health and schema compliance of input datasets."""

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_columns: List[str] = None) -> Dict[str, Any]:
        """Performs comprehensive validation checks on a dataframe."""
        validation_report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": {},
        }

        if df.empty:
            validation_report["is_valid"] = False
            validation_report["errors"].append("The dataset is completely empty.")
            return validation_report

        if expected_columns:
            missing_cols = [c for c in expected_columns if c not in df.columns]
            if missing_cols:
                validation_report["is_valid"] = False
                validation_report["errors"].append(f"Missing required columns: {missing_cols}")

        # Check for 100% null columns
        null_counts = df.isnull().sum()
        empty_cols = null_counts[null_counts == len(df)].index.tolist()
        if empty_cols:
            validation_report["warnings"].append(f"Columns completely null: {empty_cols}")

        # Check for duplicates
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            dup_pct = round((duplicate_count / len(df)) * 100, 2)
            validation_report["warnings"].append(
                f"Detected {duplicate_count} duplicate rows ({dup_pct}% of total records)."
            )

        validation_report["summary"] = {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "duplicate_rows": duplicate_count,
            "missing_cells_total": int(null_counts.sum()),
            "missing_cells_pct": round(float((null_counts.sum() / (len(df) * len(df.columns))) * 100), 2),
        }

        return validation_report
