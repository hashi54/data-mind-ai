from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.database.seed_data import seed_database
from app.rag.pipeline import RAGPipeline
from app.api.routes import (
    health,
    data,
    chat,
    predictions,
    analytics,
    documents,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("Starting DataMind AI Platform backend...")
    # 1. Ensure database is initialized & seeded
    try:
        seed_database()
    except Exception as e:
        logger.error(f"Database initialization error on startup: {e}")

    # 2. Ensure RAG knowledge base is indexed
    try:
        RAGPipeline.index_knowledge_base()
    except Exception as e:
        logger.error(f"RAG indexing error on startup: {e}")

    logger.info("DataMind AI backend is ready to serve requests.")
    yield
    logger.info("Shutting down DataMind AI backend.")


app = FastAPI(
    title="DataMind AI - API",
    description="Enterprise AI-Powered Data Intelligence, Business Analytics, SQL Agents, and ML Suite",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
origins = ["*"] if settings.ALLOWED_ORIGINS == "*" else [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api/v1
API_PREFIX = "/api/v1"
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(data.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(predictions.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)


@app.get("/")
def root():
    return {
        "platform": "DataMind AI",
        "description": "AI-Powered Data Intelligence & Business Analyst Platform",
        "version": "1.0.0",
        "documentation": "/docs",
        "api_v1": "/api/v1",
    }
