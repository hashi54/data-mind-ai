import pytest
from app.database.connection import execute_query, get_schema_summary
from app.database.models import Customer, Product, Order, Category


def test_database_tables_exist():
    """Verify all star-schema tables exist and are populated."""
    schema = get_schema_summary()
    expected_tables = ["customers", "categories", "products", "regions", "orders", "order_items", "marketing_campaigns", "customer_interactions"]
    for table in expected_tables:
        assert table in schema, f"Table '{table}' missing from database schema."


def test_database_records_populated():
    """Verify seed data contains realistic counts."""
    df_customers = execute_query("SELECT COUNT(*) AS count FROM customers")
    df_orders = execute_query("SELECT COUNT(*) AS count FROM orders")
    df_products = execute_query("SELECT COUNT(*) AS count FROM products")

    assert int(df_customers.iloc[0]["count"]) >= 100
    assert int(df_orders.iloc[0]["count"]) >= 500
    assert int(df_products.iloc[0]["count"]) >= 10


def test_analytical_revenue_query():
    """Verify analytical aggregation query executes cleanly."""
    sql = """
    SELECT 
        strftime('%Y-%m', order_date) AS month,
        SUM(total_amount) AS revenue,
        COUNT(order_id) AS orders_count
    FROM orders
    GROUP BY strftime('%Y-%m', order_date)
    ORDER BY month ASC;
    """
    df = execute_query(sql)
    assert not df.empty
    assert "revenue" in df.columns
    assert "orders_count" in df.columns
