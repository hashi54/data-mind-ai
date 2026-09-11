from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from app.core.logging import logger


class DataCleaner:
    """Automates sanitization, deduplication, and imputation on tabular data."""

    @staticmethod
    def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cleaned_df = df.copy()
        log_steps = []
        initial_rows = len(cleaned_df)

        # 1. Strip whitespace from string columns and column names
        cleaned_df.columns = [str(c).strip().replace(" ", "_").lower() for c in cleaned_df.columns]
        
        for col in cleaned_df.select_dtypes(include=["object", "string"]).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            cleaned_df[col] = cleaned_df[col].replace(["nan", "None", "NULL", "null", "N/A", ""], np.nan)

        # 2. Deduplication
        if drop_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
            dropped_dups = initial_rows - len(cleaned_df)
            if dropped_dups > 0:
                log_steps.append(f"Dropped {dropped_dups} duplicate records.")

        # 3. Numeric type inference and missing value imputation
        for col in cleaned_df.columns:
            # Try to convert to datetime if contains date-like patterns
            if "date" in col or "time" in col or "created" in col:
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
                    continue
                except Exception:
                    pass

            # Numeric columns
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                if cleaned_df[col].isnull().sum() > 0:
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    log_steps.append(f"Imputed numeric column '{col}' missing values with median ({median_val}).")
            else:
                # Categorical / string columns
                if cleaned_df[col].isnull().sum() > 0:
                    mode_val = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "Unknown"
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    log_steps.append(f"Imputed categorical column '{col}' missing values with mode ('{mode_val}').")

        summary = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(cleaned_df),
            "columns": list(cleaned_df.columns),
            "steps_applied": log_steps,
        }
        return cleaned_df, summary
