from typing import Generator
from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
from app.core.config import settings


def get_db() -> Generator[Session, None, None]:
    """Database session dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_api_key(x_api_key: str = Header(None)):
    """Optional API key security verification."""
    # In development mode, allow open access
    if settings.APP_ENV == "development":
        return True
    if x_api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return True
