import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/v1/health returns 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_loaded" in data


def test_chat_endpoint():
    """Verify POST /api/v1/chat processes business inquiries."""
    payload = {"question": "What were our sales in Kerala last month?"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "intent" in data
    assert len(data["sources"]) > 0


def test_prediction_churn_endpoint():
    """Verify POST /api/v1/predictions/churn calculates risk and SHAP."""
    payload = {"customer_id": 1}
    response = client.post("/api/v1/predictions/churn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "key_risk_factors" in data


def test_prediction_sales_endpoint():
    """Verify POST /api/v1/predictions/sales returns forecast."""
    payload = {"horizon": "30d", "model_type": "ensemble"}
    response = client.post("/api/v1/predictions/sales", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_predicted_revenue" in data
    assert len(data["forecast_data"]) == 30


def test_anomalies_endpoint():
    """Verify GET /api/v1/analytics/anomalies scans revenue anomalies."""
    response = client.get("/api/v1/analytics/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert "total_anomalies_detected" in data


def test_document_query_endpoint():
    """Verify POST /api/v1/documents/query performs RAG search."""
    payload = {"question": "What is our return policy for electronics?"}
    response = client.post("/api/v1/documents/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["retrieved_chunks"]) > 0
