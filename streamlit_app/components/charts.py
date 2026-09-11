from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Global Plotly Dark Theme Palette
DARK_THEME_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#cbd5e1", "family": "Inter, sans-serif"},
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "linecolor": "rgba(255,255,255,0.1)",
        "tickfont": {"color": "#94a3b8"},
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "linecolor": "rgba(255,255,255,0.1)",
        "tickfont": {"color": "#94a3b8"},
    },
    "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}


def plot_monthly_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Renders revenue, net revenue, and refunded amounts over time."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["month"],
        y=df["net_revenue"],
        name="Net Revenue",
        marker=dict(color="#6366f1", opacity=0.85, line=dict(color="#818cf8", width=1)),
    ))

    if "refunded_revenue" in df.columns:
        fig.add_trace(go.Bar(
            x=df["month"],
            y=df["refunded_revenue"],
            name="Refund Losses",
            marker=dict(color="#ef4444", opacity=0.75),
        ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>Monthly Net Revenue vs Refund Losses</b>",
        barmode="group",
        hovermode="x unified",
    )
    return fig


def plot_sales_forecast(forecast_data: List[Dict[str, Any]], history_df: pd.DataFrame = None) -> go.Figure:
    """Renders sales forecast with confidence interval bands."""
    df_fc = pd.DataFrame(forecast_data)
    fig = go.Figure()

    # Historical line if available
    if history_df is not None and not history_df.empty:
        fig.add_trace(go.Scatter(
            x=history_df["order_date"],
            y=history_df["revenue"],
            name="Historical Daily Sales",
            line=dict(color="#94a3b8", width=1.5),
        ))

    # Upper bound
    fig.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["upper_bound"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))

    # Lower bound + fill
    fig.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["lower_bound"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(99, 102, 241, 0.18)",
        name="95% Confidence Interval",
    ))

    # Forecast prediction line
    fig.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["forecast_revenue"],
        name="Predicted Revenue",
        line=dict(color="#38bdf8", width=3, dash="dash"),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>Forward Sales Forecast & Uncertainty Bounds</b>",
        hovermode="x unified",
    )
    return fig


def plot_regional_revenue(df: pd.DataFrame) -> go.Figure:
    """Renders horizontal bar chart of top regions by revenue."""
    fig = px.bar(
        df.sort_values(by="total_revenue", ascending=True),
        x="total_revenue",
        y="state",
        orientation="h",
        color="total_profit",
        color_continuous_scale="Viridis",
        title="<b>Revenue & Profit by State / Region</b>",
    )
    fig.update_layout(**DARK_THEME_LAYOUT)
    return fig


def plot_customer_segments_donut(segment_counts: Dict[str, int]) -> go.Figure:
    """Renders customer segment distribution donut chart."""
    labels = list(segment_counts.keys())
    values = list(segment_counts.values())
    colors = ["#8b5cf6", "#3b82f6", "#06b6d4", "#f97316", "#64748b"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        insidetextorientation="radial",
    )])
    fig.update_layout(**DARK_THEME_LAYOUT, title="<b>Customer Segment Distribution</b>")
    return fig


def plot_shap_waterfall(shap_dict: Dict[str, float]) -> go.Figure:
    """Renders SHAP feature attribution waterfall/bar chart."""
    features = list(shap_dict.keys())
    values = list(shap_dict.values())
    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>SHAP Local Feature Importance (Churn Drivers)</b>",
        xaxis_title="SHAP Value (Positive = Increases Churn Risk, Negative = Protects)",
    )
    return fig
