import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple

# -----------------------------
# RFM FEATURE ENGINEERING
# -----------------------------

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics per customer.
    """
    df = df.copy()

    # Ensure datetime format
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Snapshot date for recency calculation
    snapshot_date = df["TransactionStartTime"].max() + timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
        )
        .reset_index()
    )

    return rfm


# -----------------------------
# KMEANS CLUSTERING + RISK LABEL
# -----------------------------

def assign_risk_label(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster customers using RFM values and assign high-risk label.
    """
    rfm = rfm.copy()

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (lowest monetary value)
    high_risk_cluster = (
        rfm.groupby("cluster")["Monetary"].mean().idxmin()
    )

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm


# -----------------------------
# MERGE TARGET BACK TO DATASET
# -----------------------------

def merge_target(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Merge proxy target variable back into the main dataset.
    """
    df = df.copy()

    df = df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    # Safety: fill missing labels with low risk
    df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)

    return df


# -----------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------

def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end pipeline to create proxy credit risk target.
    """
    rfm = calculate_rfm(df)
    rfm = assign_risk_label(rfm)
    df = merge_target(df, rfm)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df

def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
