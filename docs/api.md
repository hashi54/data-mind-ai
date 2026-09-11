# DataMind AI — REST API Specification

Base URL: `http://localhost:8000/api/v1`  
Interactive OpenAPI Docs: `http://localhost:8000/docs`

---

## 1. Health & Diagnostics
### `GET /health`
Returns system status, active database dialect, and registered models.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "SQLite",
  "models_loaded": [
    "churn_logistic_regression",
    "churn_random_forest",
    "churn_gradient_boosting",
    "churn_champion",
    "sales_forecast_baseline_ridge",
    "sales_forecast_random_forest",
    "sales_forecast_gradient_boosting",
    "sales_forecast_champion",
    "customer_segmentation_kmeans",
    "customer_clv_champion"
  ],
  "vector_store_documents": 25
}
```

---

## 2. Conversational AI Business Analyst
### `POST /chat`
Accepts natural language business questions and coordinates SQL, ML, RAG, and Hybrid synthesis.

**Request:**
```json
{
  "question": "Why did sales decrease in August?"
}
```

**Response (200 OK):**
```json
{
  "answer": "Net settled revenue in August 2024 totaled ₹85,000.00, representing an 8.4% decrease compared to July...",
  "intent": "SQL_QUERY",
  "sources": ["PostgreSQL / Relational Data Warehouse"],
  "sql": "SELECT strftime('%Y-%m', order_date) AS month, COUNT(order_id) AS total_orders, SUM(CASE WHEN order_status = 'Completed' THEN total_amount ELSE 0 END) AS net_revenue...",
  "data_preview": [...],
  "chart_spec": {
    "type": "line",
    "x": "month",
    "y": "net_revenue",
    "title": "Monthly Net Revenue Trend"
  },
  "confidence": 0.98
}
```

---

## 3. Predictive Machine Learning Suite
### `POST /predictions/churn`
**Request:**
```json
{
  "customer_id": 14
}
```
**Response (200 OK):**
```json
{
  "customer_id": 14,
  "churn_probability": 0.892,
  "risk_level": "HIGH",
  "key_risk_factors": [
    "+ High number of recorded complaints (3 complaints logged)",
    "+ Long period since last purchase (78 days inactive)",
    "+ Significantly reduced purchase frequency (< 0.5 orders/month)"
  ],
  "positive_factors": [
    "- Long customer relationship tenure (16.2 months loyalty)"
  ],
  "shap_values": {
    "complaint_count": 0.35,
    "days_since_last_purchase": 0.24,
    "purchase_frequency": 0.18,
    "tenure_months": -0.22
  },
  "recommended_action": "Immediate retention outreach: Assign dedicated account manager, resolve pending tickets, and issue personalized 15% win-back incentive coupon."
}
```

### `POST /predictions/sales`
**Request:**
```json
{
  "horizon": "30d",
  "model_type": "champion"
}
```
**Response (200 OK):**
```json
{
  "horizon": "30d",
  "model_name": "sales_forecast_champion",
  "total_predicted_revenue": 1428500.0,
  "growth_rate_pct": 12.4,
  "metrics": {
    "mae": 50396.84,
    "rmse": 61084.35,
    "mape": 115.11,
    "r2_score": 0.2338
  },
  "forecast_data": [
    {
      "date": "2024-09-03",
      "forecast_revenue": 45200.0,
      "lower_bound": 0.0,
      "upper_bound": 164925.33
    }
  ],
  "insight_summary": "Projected 30d revenue is ₹1,428,500.00..."
}
```

---

## 4. Anomaly Detection & Monitoring
### `GET /analytics/anomalies`
**Response (200 OK):**
```json
{
  "total_anomalies_detected": 4,
  "anomalies": [
    {
      "date": "2024-08-03",
      "metric": "Daily Revenue",
      "actual_value": 89000.0,
      "expected_value": 24000.0,
      "deviation_pct": 270.8,
      "severity": "Warning",
      "status": "Anomaly Detected",
      "root_cause_hint": "Significant revenue surge of +270.8%. Likely promotional campaign or seasonal demand spike."
    }
  ]
}
```

---

## 5. RAG Policy Search
### `POST /documents/query`
**Request:**
```json
{
  "question": "What is the return policy for electronics?"
}
```
**Response (200 OK):**
```json
{
  "question": "What is the return policy for electronics?",
  "answer": "10-day replacement or return window for manufacturing defects, hardware failures, or transit damage...",
  "retrieved_chunks": [
    {
      "document_name": "refund_policy.md",
      "department": "Operations & Customer Experience",
      "page_number": 2,
      "similarity_score": 0.812
    }
  ]
}
```
