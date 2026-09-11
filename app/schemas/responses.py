from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    models_loaded: List[str]
    vector_store_documents: int


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: List[str]
    sql: Optional[str] = None
    data_preview: Optional[List[Dict[str, Any]]] = None
    chart_spec: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = 0.95
    limitations: Optional[str] = None


class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[int]
    churn_probability: float
    risk_level: str  # HIGH, MEDIUM, LOW
    key_risk_factors: List[str]
    positive_factors: List[str]
    shap_values: Dict[str, float]
    recommended_action: str


class ForecastDataPoint(BaseModel):
    date: str
    forecast_revenue: float
    lower_bound: float
    upper_bound: float


class SalesForecastResponse(BaseModel):
    horizon: str
    model_name: str
    total_predicted_revenue: float
    growth_rate_pct: float
    metrics: Dict[str, float]  # MAE, RMSE, MAPE, R2
    forecast_data: List[ForecastDataPoint]
    insight_summary: str


class CLVPredictionResponse(BaseModel):
    customer_id: Optional[int]
    predicted_12m_clv: float
    customer_tier: str
    avg_order_value: float
    recommended_strategy: str


class SegmentationResponse(BaseModel):
    customer_id: Optional[int]
    segment: str  # VIP, Loyal, Potential Loyalist, At Risk, Inactive
    rfm_score: Dict[str, float]
    segment_description: str
    engagement_strategy: str


class AnomalyItem(BaseModel):
    date: str
    metric: str
    actual_value: float
    expected_value: float
    deviation_pct: float
    severity: str  # Critical, Warning, Moderate
    status: str
    root_cause_hint: Optional[str] = None


class AnomalyResponse(BaseModel):
    total_anomalies_detected: int
    anomalies: List[AnomalyItem]
    scan_timestamp: str


class DocumentChunkResponse(BaseModel):
    document_name: str
    department: str
    page_number: int
    similarity_score: float
    content: str


class DocumentQueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[DocumentChunkResponse]


class EDAResponse(BaseModel):
    dataset_name: str
    total_rows: int
    total_columns: int
    column_types: Dict[str, str]
    missing_values_report: Dict[str, Any]
    duplicate_rows_count: int
    numerical_summary: Dict[str, Any]
    categorical_summary: Dict[str, Any]
    correlation_matrix: Optional[Dict[str, Any]] = None
    data_health_score: float
