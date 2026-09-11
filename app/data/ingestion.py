import os
import json
from pathlib import Path
from typing import Union, Dict, Any, Tuple
import pandas as pd
from app.core.logging import logger
from app.core.config import settings


class DataIngestionEngine:
    """Handles ingestion of raw datasets from multiple formats into Pandas DataFrames."""

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        suffix = path.suffix.lower()
        file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        
        logger.info(f"Ingesting file {path.name} ({suffix}, {file_size_mb} MB)")

        try:
            if suffix == ".csv":
                # Auto-detect separator
                try:
                    df = pd.read_csv(path, sep=None, engine="python")
                except Exception:
                    df = pd.read_csv(path)
            elif suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            elif suffix == ".json":
                df = pd.read_json(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Expected CSV, Excel, or JSON.")

            metadata = {
                "filename": path.name,
                "format": suffix,
                "file_size_mb": file_size_mb,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
            }
            return df, metadata

        except Exception as e:
            logger.error(f"Failed to ingest file {path.name}: {e}")
            raise

    @staticmethod
    def save_raw_dataset(uploaded_file_bytes: bytes, filename: str) -> Path:
        """Saves an uploaded file buffer to the raw data storage directory."""
        target_path = Path(settings.UPLOAD_DIR) / filename
        with open(target_path, "wb") as f:
            f.write(uploaded_file_bytes)
        logger.info(f"Saved uploaded raw dataset to {target_path}")
        return target_path
