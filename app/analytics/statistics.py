from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalEngine:
    """Computes descriptive and distribution statistics for analytical features."""

    @staticmethod
    def compute_numerical_summary(series: pd.Series) -> Dict[str, Any]:
        """Calculates deep descriptive statistics for a numeric pandas Series."""
        clean_s = series.dropna()
        if clean_s.empty:
            return {}

        q25 = float(np.percentile(clean_s, 25))
        q75 = float(np.percentile(clean_s, 75))
        iqr = q75 - q25

        # Outlier bounds (1.5 * IQR rule)
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_count = int(((clean_s < lower_bound) | (clean_s > upper_bound)).sum())

        skew = float(stats.skew(clean_s)) if len(clean_s) > 2 else 0.0
        kurt = float(stats.kurtosis(clean_s)) if len(clean_s) > 2 else 0.0

        return {
            "count": int(len(clean_s)),
            "mean": round(float(clean_s.mean()), 2),
            "std": round(float(clean_s.std()), 2),
            "median": round(float(clean_s.median()), 2),
            "min": round(float(clean_s.min()), 2),
            "max": round(float(clean_s.max()), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "outliers_count": outliers_count,
            "outliers_pct": round(float((outliers_count / len(clean_s)) * 100), 2),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
        }

    @staticmethod
    def compute_categorical_summary(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
        """Calculates frequency distributions for categorical series."""
        clean_s = series.dropna().astype(str)
        if clean_s.empty:
            return {}

        val_counts = clean_s.value_counts()
        total_count = len(clean_s)
        
        top_categories = {}
        for val, cnt in val_counts.head(top_n).items():
            top_categories[val] = {
                "count": int(cnt),
                "percentage": round(float((cnt / total_count) * 100), 2),
            }

        return {
            "total_count": total_count,
            "unique_count": int(clean_s.nunique()),
            "top_category": str(val_counts.index[0]) if not val_counts.empty else "N/A",
            "top_category_freq": int(val_counts.iloc[0]) if not val_counts.empty else 0,
            "distribution": top_categories,
        }
