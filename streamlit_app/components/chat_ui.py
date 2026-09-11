from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_app.components.charts import DARK_THEME_LAYOUT


def render_chat_message(message: Dict[str, Any]):
    """Renders a formatted AI or User message with citations, SQL, and Plotly charts."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                <div style="
                    background: linear-gradient(135deg, #4f46e5, #6366f1);
                    color: white;
                    border-radius: 16px 16px 4px 16px;
                    padding: 12px 18px;
                    max-width: 80%;
                    font-size: 0.95rem;
                    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
                ">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # AI Response Card
        intent = message.get("intent", "SQL_QUERY")
        sources = message.get("sources", [])
        sql = message.get("sql")
        data_preview = message.get("data_preview")
        chart_spec = message.get("chart_spec")

        # Header Badge
        intent_badges = {
            "SQL_QUERY": ("#6366f1", "SQL Analytics Agent"),
            "ML_PREDICTION": ("#ec4899", "Machine Learning Predictor"),
            "RAG_DOCS": ("#06b6d4", "Knowledge Base RAG Agent"),
            "HYBRID_SQL_RAG": ("#8b5cf6", "Hybrid RAG + SQL Orchestrator"),
            "EDA_ANALYSIS": ("#10b981", "Automated EDA Engine"),
            "GENERAL_CHAT": ("#64748b", "DataMind Assistant"),
        }
        badge_color, badge_label = intent_badges.get(intent, ("#6366f1", intent))

        with st.container():
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="
                        background: rgba(99, 102, 241, 0.15);
                        color: {badge_color};
                        border: 1px solid {badge_color};
                        font-size: 0.72rem;
                        font-weight: 700;
                        padding: 3px 8px;
                        border-radius: 6px;
                        text-transform: uppercase;
                    ">{badge_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(content)

            # Collapsible SQL query preview
            if sql:
                with st.expander("🔍 View Executed SQL Query", expanded=False):
                    st.code(sql, language="sql")

            # Data Table Preview
            if data_preview and len(data_preview) > 0:
                with st.expander("📊 View Retrieved Records", expanded=False):
                    st.dataframe(pd.DataFrame(data_preview), use_container_width=True)

            # Render Dynamic Chart if specified
            if chart_spec and data_preview:
                df_chart = pd.DataFrame(data_preview)
                ctype = chart_spec.get("type")
                if ctype == "bar" and chart_spec.get("x") in df_chart.columns and chart_spec.get("y") in df_chart.columns:
                    fig = px.bar(
                        df_chart,
                        x=chart_spec["x"],
                        y=chart_spec["y"],
                        title=f"<b>{chart_spec.get('title', 'Analysis Visualization')}</b>",
                        color_discrete_sequence=["#6366f1"],
                    )
                    fig.update_layout(**DARK_THEME_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                elif ctype == "line" and chart_spec.get("x") in df_chart.columns and chart_spec.get("y") in df_chart.columns:
                    fig = px.line(
                        df_chart,
                        x=chart_spec["x"],
                        y=chart_spec["y"],
                        title=f"<b>{chart_spec.get('title', 'Trend Analysis')}</b>",
                        markers=True,
                        color_discrete_sequence=["#38bdf8"],
                    )
                    fig.update_layout(**DARK_THEME_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            # Sources Citation Footer
            if sources:
                st.caption(f"📚 **Data & Document Sources:** {', '.join(sources)}")
                
            st.divider()
