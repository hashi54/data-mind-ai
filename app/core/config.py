import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "DataMind AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    
    # Security
    SECRET_KEY: str = "datamind-ai-super-secret-key-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: str = "*"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls", ".json", ".pdf", ".txt", ".md"]
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'datamind.db'}"
    DB_QUERY_TIMEOUT_SECONDS: int = 30
    DB_POOL_SIZE: int = 5
    
    # LLM Settings
    LLM_PROVIDER: str = "mock"  # gemini, openai, anthropic, mock
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.1
    
    # RAG & Vector Store
    VECTOR_DB_TYPE: str = "in_memory"  # in_memory, chroma
    VECTOR_DB_PATH: str = str(BASE_DIR / "data" / "vector_store")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Machine Learning
    MODEL_REGISTRY_PATH: str = str(BASE_DIR / "models")
    
    # Data Storage Paths
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "raw")
    PROCESSED_DIR: str = str(BASE_DIR / "data" / "processed")
    DOCUMENTS_DIR: str = str(BASE_DIR / "documents" / "knowledge_base")
    
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure critical directories exist
for directory in [
    BASE_DIR / "data",
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data" / "vector_store",
    BASE_DIR / "models",
    BASE_DIR / "documents" / "knowledge_base",
    BASE_DIR / "logs",
]:
    directory.mkdir(parents=True, exist_ok=True)
