# 🧠 DataMind AI — AI-Powered Data Intelligence & Business Analyst

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end AI analytics platform that combines SQL, machine learning, LLMs, RAG, automated EDA, forecasting, anomaly detection, and business intelligence to turn raw business data into actionable insights.**

---

## 🎯 What the System Does

A business user can connect or upload data and ask questions in plain English:

- *"Why did sales decrease in August?"*
- *"What were our sales in Kerala last month?"*
- *"Which customers should our sales team contact this week?"*
- *"According to our refund policy, how much revenue did we lose through refunds last month?"*
- *"Predict next month's sales revenue."*

### 🔄 Multi-Agent Decision Architecture
```
User Question
      ↓
LLM Intent Router
 ┌────┼───────────────────────────┬───────────────────────────┐
 │    ▼                           ▼                           ▼
 │  SQL Agent                 ML Agent                    RAG Agent
 │ (Read-Only)            (Model Registry)            (Vector Store)
 │    │                           │                           │
 │    ▼                           ▼                           ▼
 │ PostgreSQL / SQLite       Models: Churn,           Knowledge Base
 │ Star-Schema Warehouse     Forecast, CLV, RFM       (6 Corporate Docs)
 │    │                           │                           │
 │    ▼                           ▼                           ▼
 │ Numbers & Tables          SHAP Explainability      Relevant Chunks
 └────┴───────────────────────────┴───────────────────────────┘
                                  │
                                  ▼
                         AI Insight Generator
                        (Numbers → Language)
                                  │
                                  ▼
                    Interactive Streamlit Dashboard
```

---

## 🏗️ Core Platform Modules

### 1. 🗄️ Relational Data Warehouse (Star-Schema)
- Dual **PostgreSQL** & zero-config **SQLite** support.
- Full relational schema: `customers`, `products`, `categories`, `orders`, `order_items`, `regions`, `marketing_campaigns`, `customer_interactions`.
- Built-in enterprise data generator with 1,500+ realistic orders, seasonal spikes, intentional August revenue dips, and customer complaint histories.

### 2. 🤖 Machine Learning Model Registry
- **Customer Churn Engine**: Multi-algorithm benchmarks (Logistic Regression, Random Forest, Gradient Boosting). Evaluated on Precision, Recall, F1 (0.978), and ROC-AUC.
- **Explainable AI (TreeSHAP)**: Translates model outputs into human-readable business drivers (`+ High number of complaints`, `- Long customer tenure`).
- **Sales Forecasting**: Multi-horizon projections (7d, 30d, 90d, 1y) with 95% confidence intervals.
- **RFM Customer Segmentation**: K-Means clustering mapping customers into `VIP`, `Loyal`, `Potential Loyalist`, `At Risk`, and `Inactive`.
- **Customer Lifetime Value (CLV)**: 12-month expected forward value regression.
- **Anomaly Detection**: Dual Isolation Forest & Rolling Z-Score detector for sales spikes and sudden revenue drops.

### 3. 📚 Corporate Policy RAG & Hybrid Orchestrator
- Vector database indexing corporate knowledge base documents:
  - `refund_policy.md`
  - `employee_policy.md`
  - `product_catalog.md`
  - `sales_strategy.md`
  - `customer_support_policy.md`
  - `marketing_strategy.md`
- **Hybrid RAG + SQL**: Fuses policy text constraints with database revenue calculations in a single prompt execution.

### 4. 📊 Automated EDA Studio
- Instant upload of CSV / Excel / JSON datasets or database tables.
- Calculates **Data Health Score (0-100)**, missingness matrices, duplicate counts, outlier IQR bounds, and multivariate correlation heatmaps.

### 5. 📈 Streamlit UI (8 Dedicated Pages)
- **Executive Dashboard**: Real-time KPIs, revenue & refund trends, regional breakdown, top products.
- **AI Analyst**: Conversational interface with preset prompt chips, executed SQL inspector, and dynamic Plotly charts.
- **Customer Intelligence**: Individual customer risk scanner, SHAP waterfall cards, and retention action prescriptions.
- **Sales Forecasting**: Multi-horizon selector, confidence bands, model comparison.
- **Anomaly Detection**: Timeline anomaly scanner with severity indicators and root-cause hints.
- **Document Intelligence**: Policy search engine with highlighted citations and document uploader.
- **Automated EDA Studio**: Automated data profiling suite.
- **Model Monitoring**: Registry dashboard tracking versions, metrics (AUC/RMSE), and hyperparameters.

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/hashi54/data-mind-ai.git
cd data-mind-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
*(Optionally provide your `GEMINI_API_KEY` or `OPENAI_API_KEY`. If left empty, the built-in intelligent heuristic engine runs in offline demo mode with 100% feature coverage!)*

### 3. Initialize Database & Train Models
```bash
# Seed Database with 1,500+ records
python -m app.database.seed_data

# Train & Register ML Models
python -m app.ml.training.train_churn
python -m app.ml.training.train_forecasting
python -m app.ml.training.train_segmentation
python -m app.ml.training.train_clv

# Index Knowledge Base into Vector Store
python -m app.rag.pipeline
```

### 4. Launch Services
**Start FastAPI Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API docs available at: `http://localhost:8000/docs`*

**Start Streamlit Frontend UI:**
```bash
streamlit run streamlit_app/Home.py
```
*UI available at: `http://localhost:8501`*

---

## 🐳 Docker Deployment

To launch the complete platform (FastAPI, Streamlit, PostgreSQL, and volumes):
```bash
docker compose up --build
```

---

## 🧪 Testing

Run the comprehensive pytest suite:
```bash
pytest tests/ -v
```
All 22 unit & integration tests verify:
- API endpoint health & responses
- SQL Agent safety guardrails
- ML model training & SHAP explainability
- RAG vector retrieval & chunking
- Automated EDA profiler & anomaly detection

---

## 📁 Repository Structure

```
data-mind-ai/
├── app/
│   ├── api/routes/          # FastAPI REST endpoints
│   ├── core/                # Configuration, logging, SQL safety
│   ├── database/            # SQLAlchemy star-schema models & seed generator
│   ├── data/                # Ingestion, validation, cleaning, feature pipelines
│   ├── ml/                  # Model Registry, SHAP explainability, training & inference
│   ├── llm/                 # Multi-agent orchestrator, SQL agent, RAG router
│   ├── rag/                 # Vector store, chunking, embeddings
│   ├── analytics/           # Automated EDA engine, anomalies, KPIs
│   └── schemas/             # Pydantic request/response schemas
├── streamlit_app/
│   ├── app.py               # Main navigation shell with dark theme
│   ├── pages/               # 8 Dedicated interactive BI pages
│   └── components/          # Glassmorphic KPI cards, Plotly charts, Chat UI
├── documents/knowledge_base/ # Corporate policies & catalogs
├── sql/                     # DDL schemas & analytical cookbook
├── notebooks/               # Interactive Jupyter notebooks
├── tests/                   # 22 Pytest unit & integration tests
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
