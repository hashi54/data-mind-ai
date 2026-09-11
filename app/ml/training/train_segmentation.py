from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from app.data.preprocessing import FeatureEngineeringPipeline
from app.ml.registry import registry
from app.core.logging import logger


def train_customer_segmentation() -> Dict[str, Any]:
    """
    Trains RFM + K-Means customer segmentation clustering model.
    """
    logger.info("Extracting customer RFM features for segmentation...")
    df = FeatureEngineeringPipeline.build_customer_churn_features()

    rfm_cols = [
        "days_since_last_purchase",
        "total_orders",
        "total_spent",
        "avg_order_val",
        "tenure_months",
    ]

    X = df[rfm_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    # Compute cluster statistics to dynamically label segments
    cluster_means = df.groupby("cluster")[rfm_cols].mean()
    
    # Map clusters to business segments based on monetary spend and recency
    segment_map = {}
    sorted_by_spend = cluster_means["total_spent"].sort_values(ascending=False).index.tolist()
    
    # 0 -> VIP (highest spend), 1 -> Loyal, 2 -> Potential Loyalist, 3 -> At Risk (older recency), 4 -> Inactive
    segment_map[sorted_by_spend[0]] = "VIP"
    segment_map[sorted_by_spend[1]] = "Loyal"
    segment_map[sorted_by_spend[2]] = "Potential Loyalist"
    segment_map[sorted_by_spend[3]] = "At Risk"
    segment_map[sorted_by_spend[4]] = "Inactive"

    logger.info(f"Segment Mapping: {segment_map}")

    payload = {
        "model": kmeans,
        "scaler": scaler,
        "feature_names": rfm_cols,
        "segment_map": segment_map,
        "cluster_centers": kmeans.cluster_centers_.tolist(),
    }

    metrics = {
        "inertia": round(float(kmeans.inertia_), 2),
        "num_clusters": 5,
        "sample_count": len(df),
    }

    registry.register_model(
        model_name="customer_segmentation_kmeans",
        model_obj=payload,
        model_type="clustering",
        features=rfm_cols,
        metrics=metrics,
        parameters={"n_clusters": 5, "algorithm": "k-means++"},
    )

    return metrics


if __name__ == "__main__":
    train_customer_segmentation()
