from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="Business question or natural language query", min_length=2)
    session_id: Optional[str] = Field("default_session", description="Session identifier for multi-turn context")
    include_chart: Optional[bool] = Field(True, description="Whether to include visualization recommendations")


class PredictionChurnRequest(BaseModel):
    customer_id: Optional[int] = Field(None, description="Existing customer ID to predict")
    features: Optional[Dict[str, Any]] = Field(
        None,
        description="Explicit customer features (age, complaints, tenure_months, days_since_last_purchase, purchase_frequency, avg_order_value)",
    )


class PredictionSalesRequest(BaseModel):
    horizon: str = Field("30d", description="Forecast horizon: '7d', '30d', '90d', '1y'")
    model_type: Optional[str] = Field("ensemble", description="Model algorithm: 'ensemble', 'random_forest', 'gradient_boosting', 'linear'")


class PredictionCLVRequest(BaseModel):
    customer_id: Optional[int] = Field(None, description="Customer ID")
    features: Optional[Dict[str, Any]] = Field(None, description="Custom customer features")


class CustomerSegmentationRequest(BaseModel):
    customer_id: Optional[int] = Field(None, description="Customer ID to classify segment for")


class DocumentQueryRequest(BaseModel):
    question: str = Field(..., description="Query to search knowledge base", min_length=2)
    department: Optional[str] = Field(None, description="Optional department filter (e.g., 'Finance', 'Legal', 'Sales', 'Support')")
    top_k: Optional[int] = Field(3, description="Number of document chunks to retrieve")


class EDARequest(BaseModel):
    dataset_name: Optional[str] = Field(None, description="Name of uploaded dataset or table name")
