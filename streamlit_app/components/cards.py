import streamlit as st


def render_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal", icon: str = "📊"):
    """Renders a sleek, glassmorphic metric card with gradient glow."""
    delta_html = ""
    if delta:
        color = "#10b981" if "+" in delta or "normal" in delta_color else "#ef4444"
        delta_html = f'<div style="color: {color}; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">{delta}</div>'

    html = f"""
    <div style="
        background: rgba(18, 24, 38, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</span>
            <span style="font-size: 1.2rem;">{icon}</span>
        </div>
        <div style="color: #f8fafc; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px; font-family: 'Inter', sans-serif;">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> str:
    """Returns HTML styled badge for risk levels."""
    risk = risk_level.upper()
    if risk == "HIGH":
        bg = "rgba(239, 68, 68, 0.15)"
        border = "#ef4444"
        text = "#f87171"
    elif risk == "MEDIUM":
        bg = "rgba(245, 158, 11, 0.15)"
        border = "#f59e0b"
        text = "#fbbf24"
    else:
        bg = "rgba(16, 185, 129, 0.15)"
        border = "#10b981"
        text = "#34d399"

    return f"""
    <span style="
        background: {bg};
        color: {text};
        border: 1px solid {border};
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    ">{risk} RISK</span>
    """


def render_segment_badge(segment: str) -> str:
    """Returns styled pill badge for customer segments."""
    seg_colors = {
        "VIP": ("#8b5cf6", "rgba(139, 92, 246, 0.15)"),
        "Loyal": ("#3b82f6", "rgba(59, 130, 246, 0.15)"),
        "Potential Loyalist": ("#06b6d4", "rgba(6, 182, 212, 0.15)"),
        "At Risk": ("#f97316", "rgba(249, 115, 22, 0.15)"),
        "Inactive": ("#64748b", "rgba(100, 116, 139, 0.15)"),
    }
    border, bg = seg_colors.get(segment, ("#94a3b8", "rgba(148, 163, 184, 0.15)"))
    return f"""
    <span style="
        background: {bg};
        color: {border};
        border: 1px solid {border};
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    ">{segment}</span>
    """
