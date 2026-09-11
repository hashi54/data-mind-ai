from contextlib import contextmanager
from typing import Generator, Any, List, Dict
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.core.config import settings
from app.core.logging import logger

Base = declarative_base()

# Configure Engine based on Dialect
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for standalone database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rollback due to error: {e}")
        raise
    finally:
        session.close()


def execute_query(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Execute raw SQL query and return results as a Pandas DataFrame."""
    try:
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql), connection, params=params)
            return result_df
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise


def get_schema_summary() -> Dict[str, List[Dict[str, str]]]:
    """Inspects database tables and returns schema dictionary for LLM prompts."""
    inspector = inspect(engine)
    schema_info = {}
    
    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col.get("primary_key", False),
            })
        schema_info[table_name] = columns
        
    return schema_info
