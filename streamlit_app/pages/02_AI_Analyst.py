import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.database.seed_data import seed_database
from app.llm.analyst_agent import BusinessAnalystOrchestrator
from streamlit_app.components.chat_ui import render_chat_message

# Ensure DB is seeded (for standalone Streamlit Cloud deployment)
try:
    seed_database()
except Exception:
    pass

st.set_page_config(page_title="AI Business Analyst | DataMind AI", page_icon="🤖", layout="wide")


st.markdown("## 🤖 AI Business Analyst & Intent Orchestrator")
st.caption("Ask open-ended business questions — the platform automatically coordinates SQL agents, ML models, and policy RAG.")

# Initialize session state chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I am your **AI Business Analyst**. You can ask me questions about sales performance, customer churn, future revenue forecasts, or company policies.",
            "intent": "GENERAL_CHAT",
            "sources": ["DataMind AI Core"],
        }
    ]

# Preset Quick-Prompt Buttons
st.markdown("##### 💡 Suggested Business Questions:")
prompt_cols = st.columns(4)

preset_questions = [
    "Why did sales decrease in August?",
    "What were our sales in Kerala last month?",
    "Which customers should our sales team contact this week?",
    "According to our refund policy, how much revenue did we lose through refunds?",
]

selected_preset = None
for idx, p_text in enumerate(preset_questions):
    with prompt_cols[idx]:
        if st.button(p_text, key=f"preset_{idx}", use_container_width=True):
            selected_preset = p_text

# Render Existing Chat History
for msg in st.session_state.chat_history:
    render_chat_message(msg)

# Handle Chat Input
user_input = st.chat_input("Ask a business question (e.g. 'Predict next month's sales' or 'Why did sales fall?')...")
query_to_run = selected_preset or user_input

if query_to_run:
    # Append User Message
    st.session_state.chat_history.append({"role": "user", "content": query_to_run})
    
    with st.spinner("🤖 AI Orchestrator analyzing question across SQL, ML, and RAG..."):
        response = BusinessAnalystOrchestrator.answer_question(query_to_run)
        
        assistant_msg = {
            "role": "assistant",
            "content": response["answer"],
            "intent": response.get("intent", "SQL_QUERY"),
            "sources": response.get("sources", []),
            "sql": response.get("sql"),
            "data_preview": response.get("data_preview"),
            "chart_spec": response.get("chart_spec"),
            "confidence": response.get("confidence", 0.95),
        }
        st.session_state.chat_history.append(assistant_msg)
        
    st.rerun()
