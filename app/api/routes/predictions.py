from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.requests import (
    PredictionChurnRequest,
    PredictionSalesRequest,
    PredictionCLVRequest,
    CustomerSegmentationRequest,
)
from app.schemas.responses import (
    ChurnPredictionResponse,
    SalesForecastResponse,
    CLVPredictionResponse,
    SegmentationResponse,
)
from app.ml.inference.churn_predictor import ChurnInferenceEngine
from app.ml.inference.forecaster import SalesForecasterEngine
from app.ml.inference.clv_predictor import CLVPredictorEngine
from app.ml.inference.segmenter import CustomerSegmenterEngine
from app.database.connection import execute_query
from app.core.logging import logger

router = APIRouter(prefix="", tags=["ML Predictions & Analytics"])


@router.post("/predictions/churn", response_model=ChurnPredictionResponse)
def predict_customer_churn(req: PredictionChurnRequest):
    """Predicts customer churn probability with SHAP feature attribution drivers."""
    try:
        res = ChurnInferenceEngine.predict(
            customer_id=req.customer_id,
            custom_features=req.features,
        )
        return ChurnPredictionResponse(**res)
    except Exception as e:
        logger.error(f"Churn prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/predictions/sales", response_model=SalesForecastResponse)
def forecast_sales_revenue(req: PredictionSalesRequest):
    """Generates forward sales revenue forecast for 7d, 30d, 90d, or 1y with confidence bounds."""
    try:
        res = SalesForecasterEngine.forecast(
            horizon=req.horizon,
            model_choice=req.model_type or "champion",
        )
        return SalesForecastResponse(**res)
    except Exception as e:
        logger.error(f"Sales forecast failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/predictions/clv", response_model=CLVPredictionResponse)
def predict_customer_clv(req: PredictionCLVRequest):
    """Predicts 12-month expected forward customer value and tier."""
    try:
        res = CLVPredictorEngine.predict(
            customer_id=req.customer_id,
            custom_features=req.features,
        )
        return CLVPredictionResponse(**res)
    except Exception as e:
        logger.error(f"CLV prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/predictions/segmentation", response_model=SegmentationResponse)
def segment_customer(req: CustomerSegmentationRequest):
    """Assigns RFM behavioral segment (VIP, Loyal, Potential Loyalist, At Risk, Inactive)."""
    try:
        res = CustomerSegmenterEngine.classify(customer_id=req.customer_id)
        return SegmentationResponse(**res)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/customers/risk", response_model=List[Dict[str, Any]])
def get_high_risk_customers(limit: int = Query(10, ge=1, le=50)):
    """Fetches high-risk customers requiring immediate proactive retention outreach."""
    try:
        query = f"""
        SELECT 
            c.customer_id,
            c.name,
            c.email,
            c.customer_segment,
            c.city,
            c.state,
            COUNT(o.order_id) AS total_orders,
            COALESCE(SUM(o.total_amount), 0.0) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        WHERE c.customer_segment IN ('At Risk', 'Inactive')
        GROUP BY c.customer_id
        LIMIT {limit};
        """
        df = execute_query(query)
        results = []
        for _, row in df.iterrows():
            cid = int(row["customer_id"])
            churn_res = ChurnInferenceEngine.predict(customer_id=cid)
            results.append({
                "customer_id": cid,
                "name": row["name"],
                "email": row["email"],
                "segment": row["customer_segment"],
                "location": f"{row['city']}, {row['state']}",
                "total_orders": int(row["total_orders"]),
                "total_spent": float(row["total_spent"]),
                "churn_probability": churn_res["churn_probability"],
                "risk_level": churn_res["risk_level"],
                "key_risk_factors": churn_res["key_risk_factors"],
                "recommended_action": churn_res["recommended_action"],
            })
        return sorted(results, key=lambda x: x["churn_probability"], reverse=True)
    except Exception as e:
        logger.error(f"Error fetching customer risk: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
