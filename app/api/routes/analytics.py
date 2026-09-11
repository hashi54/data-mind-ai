from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from app.schemas.responses import AnomalyResponse
from app.analytics.anomaly import AnomalyDetectionEngine
from app.analytics.metrics import BusinessMetricsEngine
from app.database.connection import execute_query
from app.core.logging import logger

router = APIRouter(prefix="/analytics", tags=["Analytics & BI Metrics"])


@router.get("/anomalies", response_model=AnomalyResponse)
def get_revenue_anomalies():
    """Detects sales anomalies using Isolation Forest and Rolling Z-Scores."""
    try:
        res = AnomalyDetectionEngine.detect_sales_anomalies()
        return AnomalyResponse(**res)
    except Exception as e:
        logger.error(f"Error scanning anomalies: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/kpis", response_model=Dict[str, Any])
def get_executive_kpis():
    """Returns top executive KPIs (Revenue, Profit, Orders, Growth Rate)."""
    try:
        return BusinessMetricsEngine.get_executive_kpis()
    except Exception as e:
        logger.error(f"Error computing KPIs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/regional", response_model=List[Dict[str, Any]])
def get_regional_sales_breakdown():
    """Returns sales performance broken down by state and region."""
    try:
        query = """
        SELECT 
            r.region_name,
            r.state,
            COUNT(DISTINCT o.order_id) AS total_orders,
            SUM(o.total_amount) AS total_revenue,
            SUM(oi.profit) AS total_profit
        FROM regions r
        JOIN orders o ON r.region_id = o.region_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'Completed'
        GROUP BY r.region_id, r.region_name, r.state
        ORDER BY total_revenue DESC;
        """
        df = execute_query(query)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching regional breakdown: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/trends", response_model=List[Dict[str, Any]])
def get_monthly_revenue_trends():
    """Returns historical monthly revenue, orders, and refund volume."""
    try:
        query = """
        SELECT 
            strftime('%Y-%m', order_date) AS month,
            COUNT(DISTINCT order_id) AS orders_count,
            SUM(CASE WHEN order_status = 'Completed' THEN total_amount ELSE 0 END) AS net_revenue,
            SUM(CASE WHEN order_status = 'Refunded' THEN total_amount ELSE 0 END) AS refunded_revenue
        FROM orders
        GROUP BY strftime('%Y-%m', order_date)
        ORDER BY month ASC;
        """
        df = execute_query(query)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching monthly trends: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
