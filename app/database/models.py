from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Numeric,
    Date,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship
from app.database.connection import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    gender = Column(String(50))
    age = Column(Integer)
    city = Column(String(100))
    state = Column(String(100))
    signup_date = Column(Date, default=date.today)
    customer_segment = Column(String(50), default="Standard")

    # Relationships
    orders = relationship("Order", back_populates="customer", cascade="all, delete-orphan")
    interactions = relationship("CustomerInteraction", back_populates="customer", cascade="all, delete-orphan")


class Category(Base):
    __tablename__ = "categories"

    category_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category_name = Column(String(100), nullable=False, unique=True)

    # Relationships
    products = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_name = Column(String(255), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=False)
    price = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)

    # Relationships
    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")


class Region(Base):
    __tablename__ = "regions"

    region_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region_name = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    country = Column(String(100), default="India")

    # Relationships
    orders = relationship("Order", back_populates="region")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    order_date = Column(Date, nullable=False, index=True)
    region_id = Column(Integer, ForeignKey("regions.region_id"), nullable=False)
    payment_method = Column(String(50))
    order_status = Column(String(50), default="Completed")  # Completed, Refunded, Cancelled, Pending
    total_amount = Column(Float, nullable=False)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    region = relationship("Region", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Float, nullable=False)
    discount = Column(Float, default=0.0)
    profit = Column(Float, nullable=False)

    # Relationships
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")


class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"

    campaign_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    campaign_name = Column(String(255), nullable=False)
    channel = Column(String(100), nullable=False)  # Google Ads, Meta Ads, Email, SEO, Influencer
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    budget = Column(Float, nullable=False)
    conversions = Column(Integer, default=0)


class CustomerInteraction(Base):
    __tablename__ = "customer_interactions"

    interaction_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    interaction_type = Column(String(100), nullable=False)  # Support Ticket, Complaint, Inquiry, Feedback, Survey
    interaction_date = Column(DateTime, default=datetime.utcnow, index=True)
    duration = Column(Integer, default=0)  # minutes
    sentiment = Column(String(50), default="Neutral")  # Positive, Neutral, Negative
    notes = Column(Text, nullable=True)

    # Relationships
    customer = relationship("Customer", back_populates="interactions")
