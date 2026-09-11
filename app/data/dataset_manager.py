import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sqlalchemy import inspect, text
from app.core.config import settings, BASE_DIR
from app.core.logging import logger
from app.database.connection import engine, execute_query


class DatasetManager:
    """
    Manages custom company datasets uploaded by users.
    Supports CSV, Excel (.xlsx, .xls with multi-sheet support), JSON, and Parquet.
    Dynamically creates relational SQL tables and tracks active workspace datasets.
    """

    def __init__(self):
        self.meta_file = Path(settings.UPLOAD_DIR) / "datasets_registry.json"
        self._init_registry()

    def _init_registry(self):
        if not self.meta_file.exists():
            self._save_registry({"active_dataset": None, "datasets": {}})

    def _read_registry(self) -> Dict[str, Any]:
        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"active_dataset": None, "datasets": {}}

    def _save_registry(self, data: Dict[str, Any]):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_active_dataset_name(self) -> Optional[str]:
        reg = self._read_registry()
        active = reg.get("active_dataset")
        # If active exists, verify table still in DB or registry
        if active and active in reg.get("datasets", {}):
            return active
        # Fallback to first available uploaded dataset or None
        datasets = reg.get("datasets", {})
        if datasets:
            first_name = list(datasets.keys())[0]
            self.set_active_dataset(first_name)
            return first_name
        return None

    def set_active_dataset(self, dataset_name: str) -> bool:
        reg = self._read_registry()
        if dataset_name in reg.get("datasets", {}):
            reg["active_dataset"] = dataset_name
            self._save_registry(reg)
            logger.info(f"Set active dataset workspace to '{dataset_name}'")
            return True
        return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        reg = self._read_registry()
        return list(reg.get("datasets", {}).values())

    def ingest_and_register_file(
        self,
        file_bytes: bytes,
        original_filename: str,
        table_name_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parses uploaded file (CSV, Excel sheets, JSON), creates relational SQL tables,
        and registers metadata for analytical agents.
        """
        clean_stem = re.sub(r"[^\w]", "_", Path(original_filename).stem).lower()
        suffix = Path(original_filename).suffix.lower()

        saved_file_path = Path(settings.UPLOAD_DIR) / f"{clean_stem}_{int(pd.Timestamp.now().timestamp())}{suffix}"
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)

        registered_tables = []

        if suffix in [".xlsx", ".xls"]:
            # Load all sheets if multi-sheet excel
            excel_file = pd.ExcelFile(saved_file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                clean_sheet = re.sub(r"[^\w]", "_", sheet_name).lower()
                tbl_name = table_name_override or f"{clean_stem}_{clean_sheet}" if len(excel_file.sheet_names) > 1 else (table_name_override or clean_stem)
                meta = self._save_df_to_sql_table(df, tbl_name, original_filename, str(saved_file_path), sheet_name=sheet_name)
                registered_tables.append(meta)
        elif suffix == ".csv":
            try:
                df = pd.read_csv(saved_file_path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(saved_file_path)
            tbl_name = table_name_override or clean_stem
            meta = self._save_df_to_sql_table(df, tbl_name, original_filename, str(saved_file_path))
            registered_tables.append(meta)
        elif suffix == ".json":
            df = pd.read_json(saved_file_path)
            tbl_name = table_name_override or clean_stem
            meta = self._save_df_to_sql_table(df, tbl_name, original_filename, str(saved_file_path))
            registered_tables.append(meta)
        else:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: CSV, Excel (.xlsx, .xls), JSON.")

        # Set the first newly uploaded table as active
        if registered_tables:
            self.set_active_dataset(registered_tables[0]["table_name"])

        return registered_tables

    def _save_df_to_sql_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        original_filename: str,
        file_path: str,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves a Pandas DataFrame into the SQLite/Postgres database as an analytical table."""
        # Sanitize column names: lowercase, replace spaces and special chars with underscores
        clean_cols = {}
        for c in df.columns:
            clean_c = re.sub(r"[^\w]", "_", str(c).strip().lower())
            clean_cols[c] = clean_c or "col"
        df = df.rename(columns=clean_cols)
        # Deduplicate column names if any
        seen = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols

        # Detect and parse date columns
        date_cols = []
        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            clean_c = col.lower().strip()
            # Check if column name indicates date/time with word boundaries or standard suffixes
            is_date_named = bool(
                re.search(r'(^|[_\W])(date|time|timestamp|datetime|created_at|updated_at|signup_date|order_date)($|[_\W])', clean_c)
                or clean_c.endswith(("_date", "_time", "_at", "_dt", "timestamp"))
            )
            
            # If named like a date or if dtype is already datetime or string candidate
            if (is_date_named or pd.api.types.is_datetime64_any_dtype(df[col])) and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    orig_valid = df[col].notna().sum()
                    if orig_valid > 0 and (parsed.notna().sum() / orig_valid) >= 0.7:
                        df[col] = parsed
                        date_cols.append(col)
                        continue
                except Exception:
                    pass
            
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # Write to Database
        clean_table_name = re.sub(r"[^\w]", "_", table_name).lower()
        df.to_sql(clean_table_name, con=engine, if_exists="replace", index=False)
        logger.info(f"Dynamically created SQL table '{clean_table_name}' with {len(df)} rows, {len(df.columns)} columns.")

        metadata = {
            "table_name": clean_table_name,
            "original_filename": original_filename,
            "sheet_name": sheet_name,
            "file_path": file_path,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "date_columns": date_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "uploaded_at": pd.Timestamp.now().isoformat(),
        }

        # Update Registry
        reg = self._read_registry()
        reg["datasets"][clean_table_name] = metadata
        self._save_registry(reg)

        return metadata

    def get_dataset_df(self, table_name: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
        target_table = table_name or self.get_active_dataset_name()
        if not target_table:
            # Fallback to orders if exists or empty dataframe
            target_table = "orders"
        try:
            return execute_query(f"SELECT * FROM {target_table} LIMIT {limit};")
        except Exception as e:
            logger.warning(f"Could not load table '{target_table}': {e}")
            return pd.DataFrame()

    def get_all_table_names(self) -> List[str]:
        """Returns all tables present in the database."""
        inspector = inspect(engine)
        return inspector.get_table_names()

    def delete_dataset(self, table_name: str) -> bool:
        """Drops dynamic table and removes metadata from registry."""
        try:
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
                conn.commit()
            reg = self._read_registry()
            if table_name in reg.get("datasets", {}):
                del reg["datasets"][table_name]
                if reg.get("active_dataset") == table_name:
                    reg["active_dataset"] = list(reg["datasets"].keys())[0] if reg["datasets"] else None
                self._save_registry(reg)
            logger.info(f"Dropped table and removed dataset '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Error dropping table '{table_name}': {e}")
            return False


dataset_manager = DatasetManager()
