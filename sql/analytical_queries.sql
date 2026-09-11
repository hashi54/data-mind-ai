-- ==========================================================
-- DataMind AI - Benchmark Analytical SQL Queries
-- ==========================================================

-- 1. Monthly Revenue & Order Volume Trends
SELECT 
    strftime('%Y-%m', order_date) AS month,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(total_amount) AS gross_revenue,
    SUM(CASE WHEN order_status = 'Completed' THEN total_amount ELSE 0 END) AS net_revenue,
    SUM(CASE WHEN order_status = 'Refunded' THEN total_amount ELSE 0 END) AS refunded_revenue
FROM orders
GROUP BY strftime('%Y-%m', order_date)
ORDER BY month ASC;

-- 2. Regional Sales & Profit Distribution (e.g., Kerala Performance)
SELECT 
    r.region_name,
    r.state,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.total_amount) AS total_sales,
    SUM(oi.profit) AS total_profit,
    ROUND(SUM(oi.profit) * 100.0 / NULLIF(SUM(o.total_amount), 0), 2) AS profit_margin_pct
FROM regions r
JOIN orders o ON r.region_id = o.region_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_status = 'Completed'
GROUP BY r.region_id, r.region_name, r.state
ORDER BY total_sales DESC;

-- 3. Top Selling Products and Gross Profitability
SELECT 
    p.product_id,
    p.product_name,
    c.category_name,
    SUM(oi.quantity) AS total_units_sold,
    SUM(oi.unit_price * oi.quantity) AS total_revenue,
    SUM(oi.profit) AS total_profit
FROM products p
JOIN categories c ON p.category_id = c.category_id
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'Completed'
GROUP BY p.product_id, p.product_name, c.category_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Customer RFM Feature Extraction
WITH customer_orders AS (
    SELECT 
        c.customer_id,
        c.name,
        c.email,
        c.signup_date,
        MAX(o.order_date) AS last_order_date,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary_value,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status = 'Completed'
    GROUP BY c.customer_id, c.name, c.email, c.signup_date
),
customer_complaints AS (
    SELECT 
        customer_id,
        COUNT(interaction_id) AS total_complaints
    FROM customer_interactions
    WHERE interaction_type IN ('Complaint', 'Return Request')
    GROUP BY customer_id
)
SELECT 
    co.customer_id,
    co.name,
    co.email,
    co.frequency,
    COALESCE(co.monetary_value, 0) AS total_spend,
    COALESCE(co.avg_order_value, 0) AS aov,
    COALESCE(cc.total_complaints, 0) AS complaints_count
FROM customer_orders co
LEFT JOIN customer_complaints cc ON co.customer_id = cc.customer_id
ORDER BY total_spend DESC;

-- 5. Refund Revenue Loss Analysis
SELECT 
    strftime('%Y-%m', order_date) AS month,
    COUNT(order_id) AS refunded_orders_count,
    SUM(total_amount) AS lost_refund_revenue
FROM orders
WHERE order_status = 'Refunded'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY month DESC;
