from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.responses import EDAResponse
from app.data.ingestion import DataIngestionEngine
from app.data.cleaning import DataCleaner
from app.analytics.eda import AutomatedEDAEngine
from app.core.security import sanitize_filename
from app.core.logging import logger

router = APIRouter(prefix="/data", tags=["Data Ingestion & Automated EDA"])


@router.post("/upload", response_model=Dict[str, Any])
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a CSV, Excel, or JSON dataset for processing and automated profiling."""
    safe_name = sanitize_filename(file.filename)
    contents = await file.read()
    
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File exceeds 50MB limit.")

    try:
        saved_path = DataIngestionEngine.save_raw_dataset(contents, safe_name)
        df, meta = DataIngestionEngine.load_file(saved_path)
        return {
            "status": "success",
            "message": f"Successfully uploaded and ingested {safe_name}",
            "metadata": meta,
        }
    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{dataset_name}/profile", response_model=EDAResponse)
def profile_dataset(dataset_name: str):
    """Generates deep automated exploratory data analysis (EDA) profile."""
    from app.core.config import settings
    file_path = Path(settings.UPLOAD_DIR) / dataset_name

    # If dataset not in raw uploads, check sample data or database
    if not file_path.exists():
        # Fallback to loading from orders table
        from app.database.connection import execute_query
        try:
            df = execute_query("SELECT * FROM orders LIMIT 1000;")
            profile = AutomatedEDAEngine.profile_dataframe(df, dataset_name="Orders Warehouse Table")
            return EDAResponse(**profile)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {dataset_name} not found.")

    df, _ = DataIngestionEngine.load_file(file_path)
    profile = AutomatedEDAEngine.profile_dataframe(df, dataset_name=dataset_name)
    return EDAResponse(**profile)
