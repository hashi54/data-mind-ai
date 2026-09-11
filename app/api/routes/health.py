from fastapi import APIRouter, Depends
from app.schemas.responses import HealthResponse
from app.ml.registry import registry
from app.rag.vector_store import vector_store
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", response_model=HealthResponse)
def health_check():
    """Returns system status, active database dialect, and loaded ML models."""
    models = list(registry.list_all_models().keys())
    db_type = "PostgreSQL" if "postgres" in settings.DATABASE_URL.lower() else "SQLite"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database=db_type,
        models_loaded=models,
        vector_store_documents=vector_store.get_document_count(),
    )
