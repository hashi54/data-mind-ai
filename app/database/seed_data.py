import random
from datetime import datetime, date, timedelta
from app.database.connection import engine, Base, SessionLocal
from app.database.models import (
    Customer,
    Category,
    Product,
    Region,
    Order,
    OrderItem,
    MarketingCampaign,
    CustomerInteraction,
)
from app.core.logging import logger


def init_db():
    """Create all database tables."""
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def seed_database():
    """Seeds the database with realistic enterprise data."""
    init_db()
    session = SessionLocal()

    # Check if data already exists
    if session.query(Customer).count() > 50:
        logger.info("Database already contains seed data. Skipping seed generation.")
        session.close()
        return

    logger.info("Generating realistic enterprise seed data...")
    random.seed(42)

    # 1. Categories
    category_names = [
        "Consumer Electronics",
        "Home & Kitchen Appliances",
        "Furniture & Living",
        "Fashion & Apparel",
        "Footwear & Athleisure",
        "Beauty & Personal Care",
        "Sports & Fitness Gear",
        "Smart Office & Computing",
        "Gourmet & Organic Grocery",
        "Audio & Wearables",
    ]
    categories = []
    for name in category_names:
        cat = Category(category_name=name)
        session.add(cat)
        categories.append(cat)
    session.flush()

    # 2. Products
    product_blueprints = [
        # Electronics & Computing
        ("UltraView 4K Smart TV 55-inch", 0, 48000.0, 32000.0, 150),
        ("AeroBook Pro 15 Laptop (16GB RAM)", 7, 72000.0, 52000.0, 80),
        ("NoiseCancelling Pro Over-Ear Headphones", 9, 14999.0, 8500.0, 240),
        ("SmartWatch Series 5 GPS", 9, 18999.0, 11000.0, 300),
        ("OptiCharge 65W GaN Fast Charger", 0, 2499.0, 950.0, 500),
        ("Titan Mechanical Gaming Keyboard", 7, 6499.0, 3100.0, 180),
        ("ErgoGrip Wireless Vertical Mouse", 7, 2999.0, 1200.0, 350),
        ("CrystalBass Portable Bluetooth Speaker", 9, 4999.0, 2200.0, 400),
        # Home & Appliances
        ("SilentAir Dual-Inverter AC 1.5 Ton", 1, 38500.0, 26000.0, 90),
        ("PureFlow RO+UV Water Purifier", 1, 16500.0, 9800.0, 120),
        ("QuickGrind 750W Mixer Grinder", 1, 3800.0, 2100.0, 220),
        ("ChefMaster Digital Air Fryer 5L", 1, 7499.0, 4200.0, 160),
        # Furniture
        ("ErgoLux Executive Ergonomic Chair", 2, 14500.0, 8200.0, 75),
        ("Nordic Solid Oak Study Desk", 2, 22000.0, 13500.0, 40),
        ("VelvetComfort 3-Seater Living Sofa", 2, 36000.0, 21000.0, 30),
        ("Minimalist Floating Bookshelf", 2, 4500.0, 2200.0, 110),
        # Fashion & Footwear
        ("Classic Oxford Cotton Shirt", 3, 2499.0, 850.0, 300),
        ("Slim-Fit Raw Selvedge Denim Jeans", 3, 3499.0, 1400.0, 250),
        ("All-Weather Waterproof Trench Coat", 3, 6999.0, 3100.0, 90),
        ("CloudStrider Performance Running Shoes", 4, 5999.0, 2600.0, 200),
        ("UrbanLeather Chelsea Boots", 4, 6499.0, 2900.0, 140),
        ("Everyday Canvas Low-Top Sneakers", 4, 2899.0, 1100.0, 320),
        # Beauty, Sports & Grocery
        ("HydraGlow Vitamin C Serum 30ml", 5, 1299.0, 320.0, 600),
        ("Botanical Repair Hair Mask", 5, 899.0, 240.0, 450),
        ("ProFit Adjustable Dumbbell Set 24kg", 6, 12500.0, 6800.0, 85),
        ("EcoGrip Natural Rubber Yoga Mat", 6, 2199.0, 750.0, 280),
        ("Organic Single-Origin Filter Coffee 500g", 8, 650.0, 280.0, 500),
        ("Himalayan Wild Forest Raw Honey 500g", 8, 750.0, 310.0, 420),
    ]
    products = []
    for name, cat_idx, price, cost, stock in product_blueprints:
        prod = Product(
            product_name=name,
            category_id=categories[cat_idx].category_id,
            price=price,
            cost=cost,
            stock_quantity=stock,
        )
        session.add(prod)
        products.append(prod)
    session.flush()

    # 3. Regions
    region_data = [
        ("South Region - Kerala", "Kerala", "India"),
        ("West Region - Maharashtra", "Maharashtra", "India"),
        ("South Region - Karnataka", "Karnataka", "India"),
        ("South Region - Tamil Nadu", "Tamil Nadu", "India"),
        ("North Region - Delhi NCR", "Delhi", "India"),
        ("South Region - Telangana", "Telangana", "India"),
        ("West Region - Gujarat", "Gujarat", "India"),
        ("North Region - Uttar Pradesh", "Uttar Pradesh", "India"),
        ("East Region - West Bengal", "West Bengal", "India"),
        ("North Region - Rajasthan", "Rajasthan", "India"),
        ("Central Region - Madhya Pradesh", "Madhya Pradesh", "India"),
        ("North Region - Punjab", "Punjab", "India"),
    ]
    regions = []
    for reg_name, state_name, country in region_data:
        reg = Region(region_name=reg_name, state=state_name, country=country)
        session.add(reg)
        regions.append(reg)
    session.flush()

    # 4. Customers
    first_names = [
        "Aarav", "Aditi", "Rohan", "Priya", "Vikram", "Sneha", "Rahul", "Ananya", "Karan", "Pooja",
        "Arjun", "Divya", "Siddharth", "Neha", "Varun", "Meera", "Kabir", "Rhea", "Nikhil", "Ishita",
        "Aditya", "Tanvi", "Gaurav", "Simran", "Amit", "Kavita", "Suresh", "Lakshmi", "Rajesh", "Sunita",
        "Deepak", "Swati", "Manoj", "Shweta", "Anand", "Pallavi", "Harish", "Archana", "Sanjay", "Geeta",
    ]
    last_names = [
        "Sharma", "Verma", "Patel", "Nair", "Menon", "Reddy", "Rao", "Gupta", "Iyer", "Kulkarni",
        "Deshmukh", "Kapoor", "Malhotra", "Mehta", "Bose", "Chatterjee", "Singh", "Kaur", "Chopra", "Das",
        "Pillai", "Kurian", "Gowda", "Hegde", "Shetty", "Bhat", "Joshi", "Mishra", "Pandey", "Saxena",
    ]
    genders = ["Male", "Female", "Other"]
    cities_states = [
        ("Kochi", "Kerala"), ("Thiruvananthapuram", "Kerala"), ("Kozhikode", "Kerala"),
        ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Nagpur", "Maharashtra"),
        ("Bengaluru", "Karnataka"), ("Mysuru", "Karnataka"), ("Hubballi", "Karnataka"),
        ("Chennai", "Tamil Nadu"), ("Coimbatore", "Tamil Nadu"), ("Madurai", "Tamil Nadu"),
        ("New Delhi", "Delhi"), ("Noida", "Uttar Pradesh"), ("Gurugram", "Haryana"),
        ("Hyderabad", "Telangana"), ("Warangal", "Telangana"),
        ("Ahmedabad", "Gujarat"), ("Surat", "Gujarat"), ("Vadodara", "Gujarat"),
        ("Kolkata", "West Bengal"), ("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh"),
    ]

    customers = []
    base_start_date = date.today() - timedelta(days=730)
    
    for i in range(1, 351):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name = f"{fn} {ln}"
        email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
        gender = random.choice(genders)
        age = random.randint(21, 68)
        city, state = random.choice(cities_states)
        signup_dt = base_start_date + timedelta(days=random.randint(0, 680))
        
        # Segment pre-allocation
        seg_choices = ["VIP", "Loyal", "Potential Loyalist", "Standard", "At Risk", "Inactive"]
        weights = [0.10, 0.20, 0.25, 0.25, 0.12, 0.08]
        segment = random.choices(seg_choices, weights=weights)[0]

        cust = Customer(
            name=name,
            email=email,
            gender=gender,
            age=age,
            city=city,
            state=state,
            signup_date=signup_dt,
            customer_segment=segment,
        )
        session.add(cust)
        customers.append(cust)
    session.flush()

    # 5. Marketing Campaigns
    campaign_defs = [
        ("Diwali Mega Fest 2024", "Meta Ads", date(2024, 10, 15), date(2024, 11, 5), 450000.0, 1850),
        ("Great Summer Electronics Sale", "Google Ads", date(2024, 5, 1), date(2024, 5, 20), 380000.0, 1420),
        ("Kerala Onam Special Promotion", "Meta Ads", date(2024, 9, 1), date(2024, 9, 16), 250000.0, 1100),
        ("Monsoon Home Makeover", "Email", date(2024, 7, 10), date(2024, 7, 31), 85000.0, 620),
        ("Independence Day Flash Deals", "Google Ads", date(2024, 8, 10), date(2024, 8, 16), 180000.0, 780),
        ("Year-End Tech Clearance", "Google Ads", date(2024, 12, 20), date(2025, 1, 5), 520000.0, 2100),
        ("Spring Fitness & Wellness", "Influencer", date(2025, 2, 1), date(2025, 2, 28), 160000.0, 590),
        ("Republic Day Special Sale", "Meta Ads", date(2025, 1, 20), date(2025, 1, 28), 220000.0, 940),
        ("Organic Living Awareness", "SEO", date(2024, 3, 1), date(2024, 6, 30), 110000.0, 890),
        ("Back to School Tech Bundles", "Google Ads", date(2024, 6, 1), date(2024, 6, 25), 290000.0, 1150),
    ]
    for cname, chan, sdt, edt, bud, conv in campaign_defs:
        camp = MarketingCampaign(
            campaign_name=cname,
            channel=chan,
            start_date=sdt,
            end_date=edt,
            budget=bud,
            conversions=conv,
        )
        session.add(camp)
    session.flush()

    # 6. Orders and Order Items
    payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]
    order_statuses = ["Completed", "Completed", "Completed", "Completed", "Refunded", "Cancelled"]

    orders = []
    order_items = []
    
    # Generate daily order flow over past 550 days
    start_history = date.today() - timedelta(days=550)
    current_dt = start_history

    while current_dt <= date.today():
        # Day-of-week factor (weekends higher)
        is_weekend = current_dt.weekday() in [5, 6]
        base_orders = random.randint(4, 8) if is_weekend else random.randint(2, 5)

        # Intentional August 2024 dip (simulating supply chain crunch and ad spend reduction)
        if current_dt.year == 2024 and current_dt.month == 8:
            base_orders = max(1, int(base_orders * 0.65))

        # Peak festive season (October / November Diwali boost)
        if current_dt.year == 2024 and current_dt.month in [10, 11]:
            base_orders = int(base_orders * 1.5)

        for _ in range(base_orders):
            cust = random.choice(customers)
            # Ensure order date is after customer signup date
            if cust.signup_date > current_dt:
                continue

            # Region mapped from customer state
            matching_regions = [r for r in regions if r.state == cust.state]
            reg = matching_regions[0] if matching_regions else random.choice(regions)

            # Order Status
            status = random.choices(
                ["Completed", "Refunded", "Cancelled"],
                weights=[0.88, 0.08, 0.04] if not (current_dt.year == 2024 and current_dt.month == 8) else [0.75, 0.18, 0.07],
            )[0]

            pay_method = random.choice(payment_methods)

            # Order Items (1 to 4 items per order)
            num_items = random.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.12, 0.05])[0]
            selected_products = random.sample(products, num_items)

            order_total = 0.0
            temp_items = []

            for prod in selected_products:
                qty = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
                discount_pct = random.choice([0.0, 0.05, 0.10, 0.15, 0.20])
                effective_unit_price = round(prod.price * (1 - discount_pct), 2)
                item_revenue = effective_unit_price * qty
                item_cost = prod.cost * qty
                item_profit = round(item_revenue - item_cost, 2)
                order_total += item_revenue

                temp_items.append({
                    "product_id": prod.product_id,
                    "quantity": qty,
                    "unit_price": effective_unit_price,
                    "discount": discount_pct,
                    "profit": item_profit,
                })

            order = Order(
                customer_id=cust.customer_id,
                order_date=current_dt,
                region_id=reg.region_id,
                payment_method=pay_method,
                order_status=status,
                total_amount=round(order_total, 2),
            )
            session.add(order)
            session.flush()

            for item_data in temp_items:
                oi = OrderItem(
                    order_id=order.order_id,
                    product_id=item_data["product_id"],
                    quantity=item_data["quantity"],
                    unit_price=item_data["unit_price"],
                    discount=item_data["discount"],
                    profit=item_data["profit"],
                )
                session.add(oi)

        current_dt += timedelta(days=1)

    session.flush()

    # 7. Customer Interactions (Complaints, Tickets, Surveys)
    interaction_types = ["Support Ticket", "Complaint", "Product Inquiry", "Feedback", "Return Request"]
    sentiments = ["Positive", "Neutral", "Negative"]

    for cust in customers:
        # At-risk and Inactive customers have more complaints and negative sentiment
        is_at_risk = cust.customer_segment in ["At Risk", "Inactive"]
        num_interactions = random.randint(3, 7) if is_at_risk else random.randint(1, 3)

        for _ in range(num_interactions):
            itype = random.choices(
                interaction_types,
                weights=[0.15, 0.50, 0.10, 0.10, 0.15] if is_at_risk else [0.35, 0.10, 0.30, 0.20, 0.05],
            )[0]
            sent = random.choices(
                sentiments,
                weights=[0.1, 0.2, 0.7] if is_at_risk else [0.6, 0.3, 0.1],
            )[0]
            days_ago = random.randint(1, 180)
            idate = datetime.utcnow() - timedelta(days=days_ago, minutes=random.randint(10, 500))
            duration = random.randint(2, 45)

            note_templates = {
                "Complaint": [
                    "Customer reported delayed delivery and poor package condition.",
                    "App crashed during checkout and payment was debited twice.",
                    "Product stopped working after 2 weeks, requested replacement.",
                    "Customer service call was disconnected without resolution.",
                ],
                "Return Request": [
                    "Requested return as item size did not match specifications.",
                    "Defective display on received smart device, requested refund under 14-day policy.",
                    "Ordered wrong variant by mistake, processed return.",
                ],
                "Support Ticket": [
                    "Inquired about warranty extension policy and AMC charges.",
                    "Requested invoice copy for tax filing.",
                    "Asked about upcoming festive discounts on electronics.",
                ],
                "Feedback": [
                    "Extremely satisfied with product build quality and fast courier.",
                    "Good product but user manual could be clearer.",
                ],
                "Product Inquiry": [
                    "Checked compatibility with Mac and Windows systems.",
                    "Inquired about bulk corporate purchasing options.",
                ],
            }
            notes = random.choice(note_templates.get(itype, ["Customer inquiry recorded."]))

            ci = CustomerInteraction(
                customer_id=cust.customer_id,
                interaction_type=itype,
                interaction_date=idate,
                duration=duration,
                sentiment=sent,
                notes=notes,
            )
            session.add(ci)

    session.commit()
    logger.info(f"Database seeded successfully with {len(customers)} customers, {len(products)} products, {session.query(Order).count()} orders!")
    session.close()


if __name__ == "__main__":
    seed_database()
