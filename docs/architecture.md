# DataMind AI — Architecture & System Design Document

DataMind AI is an enterprise AI analytics platform combining natural language orchestration, SQL agents, ML model registries, Explainable AI (SHAP), corporate policy RAG, automated EDA, and interactive business intelligence.

---

## 1. High-Level System Architecture

```
                             ┌──────────────────────┐
                             │       USER           │
                             │ Manager / Analyst    │
                             └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │   Streamlit App UI   │
                             │ (8 Dedicated Pages)  │
                             └──────────┬───────────┘
                                        │ (HTTP / REST)
                                        ▼
                             ┌──────────────────────┐
                             │   FastAPI Backend    │
                             │  (/api/v1 Endpoints) │
                             └──────────┬───────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       AI ORCHESTRATOR        │
                         │   (Intent Router & Agents)   │
                         └──────────────┬───────────────┘
                                        │
                 ┌──────────────────────┼─────────────────────┐
                 │                      │                     │
                 ▼                      ▼                     ▼
          ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
          │  SQL Agent  │       │  ML Agent   │       │  RAG Agent  │
          │ (Read-Only) │       │ (5 Models)  │       │ (Vector DB) │
          └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                 │                     │                     │
                 ▼                     ▼                     ▼
          ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
          │ PostgreSQL  │       │ Model Reg.  │       │ Knowledge   │
          │ / SQLite DB │       │ (Joblib/SHAP│       │ Base Chunks │
          └─────────────┘       └─────────────┘       └─────────────┘
                 │                     │                     │
                 └─────────────────────┼─────────────────────┘
                                       ▼
                             ┌──────────────────────┐
                             │  Insight Generator   │
                             │ (Numbers -> Language)│
                             └──────────┬───────────┘
                                        ▼
                             ┌──────────────────────┐
                             │ Interactive Dashboard│
                             │   Plotly Visuals     │
                             └──────────────────────┘
```

---

## 2. Intent Routing & Multi-Agent Decisions

1. **`SQL_QUERY` Intent**:
   - User asks for aggregations, counts, totals, or data filtering.
   - SQL Agent validates read-only compliance, runs dialect query on PostgreSQL / SQLite, and generates natural language explanations with Plotly specs.

2. **`ML_PREDICTION` Intent**:
   - **Churn**: Loads champion model (`churn_random_forest`, F1: 0.98), computes probability, and derives SHAP feature waterfall drivers.
   - **Sales Forecasting**: Multi-horizon engine projecting forward daily/weekly/monthly revenue with 95% confidence intervals.
   - **Customer Lifetime Value (CLV)**: 12-month expected customer value regression.
   - **Customer Segmentation**: RFM + K-Means clustering (VIP, Loyal, Potential Loyalist, At Risk, Inactive).

3. **`RAG_DOCS` Intent**:
   - Semantic retrieval against company policy documents (Refund, HR, Product Catalog, Sales Strategy, Customer Support, Marketing Strategy).

4. **`HYBRID_SQL_RAG` Intent**:
   - Decomposes inquiries requiring both policy definitions and live data calculations.
   - Example: *"According to our refund policy, how much revenue did we lose through refunds last month?"*
   - Fetches refund policy SLAs from RAG and queries refunded order revenue from SQL, then synthesizes an executive summary.

5. **`EDA_ANALYSIS` Intent**:
   - Automated profiling of tabular datasets: schema inference, missingness reports, outlier detection, and correlation matrices.
