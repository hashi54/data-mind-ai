# DataMind AI — Star-Schema Database Documentation

DataMind AI is powered by a star-schema analytics database designed for realistic enterprise workloads.

---

## 1. Entity-Relationship Schema

### `categories` (Dimension Table)
| Column Name | Type | Description |
|---|---|---|
| `category_id` | SERIAL PK | Unique identifier for product category |
| `category_name` | VARCHAR(100) | Name of category (e.g. Consumer Electronics) |

### `products` (Dimension Table)
| Column Name | Type | Description |
|---|---|---|
| `product_id` | SERIAL PK | Unique product identifier |
| `product_name` | VARCHAR(255) | Full product title |
| `category_id` | INTEGER FK | Foreign key referencing categories |
| `price` | NUMERIC(12, 2) | Product retail selling price (₹) |
| `cost` | NUMERIC(12, 2) | Product procurement / manufacturing cost (₹) |
| `stock_quantity` | INTEGER | Current available warehouse stock |

### `regions` (Dimension Table)
| Column Name | Type | Description |
|---|---|---|
| `region_id` | SERIAL PK | Unique territory identifier |
| `region_name` | VARCHAR(100) | Territory name (e.g., South Region - Kerala) |
| `state` | VARCHAR(100) | State / Province |
| `country` | VARCHAR(100) | Country (Default: India) |

### `customers` (Dimension Table)
| Column Name | Type | Description |
|---|---|---|
| `customer_id` | SERIAL PK | Customer identifier |
| `name` | VARCHAR(255) | Full name |
| `email` | VARCHAR(255) | Unique email |
| `gender` | VARCHAR(50) | Gender |
| `age` | INTEGER | Customer age |
| `city` | VARCHAR(100) | City of residence |
| `state` | VARCHAR(100) | State |
| `signup_date` | DATE | Account registration date |
| `customer_segment` | VARCHAR(50) | Assigned segment (VIP, Loyal, At Risk, etc.) |

### `orders` (Fact Table)
| Column Name | Type | Description |
|---|---|---|
| `order_id` | SERIAL PK | Transaction identifier |
| `customer_id` | INTEGER FK | Foreign key referencing customers |
| `order_date` | DATE | Transaction settlement date |
| `region_id` | INTEGER FK | Delivery territory FK |
| `payment_method` | VARCHAR(50) | UPI, Credit Card, Net Banking, COD |
| `order_status` | VARCHAR(50) | Completed, Refunded, Cancelled |
| `total_amount` | NUMERIC(12, 2) | Gross order value (₹) |

### `order_items` (Fact Line Items Table)
| Column Name | Type | Description |
|---|---|---|
| `order_item_id` | SERIAL PK | Line item identifier |
| `order_id` | INTEGER FK | FK to orders |
| `product_id` | INTEGER FK | FK to products |
| `quantity` | INTEGER | Units purchased |
| `unit_price` | NUMERIC(12, 2) | Effective discounted unit price |
| `discount` | NUMERIC(5, 2) | Applied discount percentage |
| `profit` | NUMERIC(12, 2) | Net gross profit for line item |

### `marketing_campaigns` (Dimension Table)
| Column Name | Type | Description |
|---|---|---|
| `campaign_id` | SERIAL PK | Campaign identifier |
| `campaign_name` | VARCHAR(255) | Name of campaign (e.g., Diwali Mega Fest) |
| `channel` | VARCHAR(100) | Channel (Meta Ads, Google Ads, Email, SEO) |
| `start_date` | DATE | Launch date |
| `end_date` | DATE | Conclusion date |
| `budget` | NUMERIC(12, 2) | Total marketing budget allocated |
| `conversions` | INTEGER | Attributed customer acquisitions |

### `customer_interactions` (Event Fact Table)
| Column Name | Type | Description |
|---|---|---|
| `interaction_id` | SERIAL PK | Event identifier |
| `customer_id` | INTEGER FK | Customer FK |
| `interaction_type` | VARCHAR(100) | Support Ticket, Complaint, Inquiry |
| `interaction_date` | TIMESTAMP | Timestamp of interaction |
| `duration` | INTEGER | Call / resolution duration (minutes) |
| `sentiment` | VARCHAR(50) | Positive, Neutral, Negative |
| `notes` | TEXT | Detailed agent notes |
