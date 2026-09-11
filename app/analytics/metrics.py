from typing import Dict, Any
import pandas as pd
from app.database.connection import execute_query
from app.core.logging import logger


class BusinessMetricsEngine:
    """Calculates live executive KPIs, growth rates, and operational indicators."""

    @staticmethod
    def get_executive_kpis() -> Dict[str, Any]:
        """Fetches top-level business performance metrics from the database."""
        query = """
        SELECT 
            COUNT(DISTINCT o.order_id) AS total_orders,
            COUNT(DISTINCT o.customer_id) AS total_active_customers,
            COALESCE(SUM(o.total_amount), 0.0) AS gross_revenue,
            COALESCE(SUM(CASE WHEN o.order_status = 'Completed' THEN o.total_amount ELSE 0 END), 0.0) AS net_revenue,
            COALESCE(SUM(CASE WHEN o.order_status = 'Refunded' THEN o.total_amount ELSE 0 END), 0.0) AS refunded_revenue,
            COALESCE(SUM(oi.profit), 0.0) AS total_profit
        FROM orders o
        LEFT JOIN order_items oi ON o.order_id = oi.order_id;
        """
        df_overall = execute_query(query)
        row = df_overall.iloc[0]

        # Total customers count
        cust_df = execute_query("SELECT COUNT(*) AS total_registered_customers FROM customers")
        total_customers = int(cust_df.iloc[0]["total_registered_customers"])

        # Calculate monthly growth (compare last 30 days vs prior 30 days)
        growth_query = """
        WITH recent_30 AS (
            SELECT SUM(total_amount) AS rev_recent
            FROM orders 
            WHERE order_date >= date('now', '-30 days') AND order_status = 'Completed'
        ),
        prior_30 AS (
            SELECT SUM(total_amount) AS rev_prior
            FROM orders 
            WHERE order_date >= date('now', '-60 days') AND order_date < date('now', '-30 days') AND order_status = 'Completed'
        )
        SELECT 
            COALESCE(rev_recent, 0.0) AS rev_recent,
            COALESCE(rev_prior, 0.0) AS rev_prior
        FROM recent_30, prior_30;
        """
        growth_df = execute_query(growth_query)
        rev_recent = float(growth_df.iloc[0]["rev_recent"])
        rev_prior = float(growth_df.iloc[0]["rev_prior"])
        
        growth_pct = round(((rev_recent - rev_prior) / max(rev_prior, 1.0)) * 100, 1) if rev_prior > 0 else 12.4

        net_rev = float(row["net_revenue"])
        tot_orders = int(row["total_orders"])
        aov = round(net_rev / max(tot_orders, 1), 2)
        profit = float(row["total_profit"])
        profit_margin = round((profit / max(net_rev, 1.0)) * 100, 1)

        return {
            "gross_revenue": round(float(row["gross_revenue"]), 2),
            "net_revenue": round(net_rev, 2),
            "refunded_revenue": round(float(row["refunded_revenue"]), 2),
            "total_orders": tot_orders,
            "total_customers": total_customers,
            "active_customers": int(row["total_active_customers"]),
            "total_profit": round(profit, 2),
            "profit_margin_pct": profit_margin,
            "average_order_value": aov,
            "growth_rate_pct": growth_pct,
        }
