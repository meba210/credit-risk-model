import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

##Calculate RFM Metrics

def calculate_rfm(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    snapshot_date = df["TransactionStartTime"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Value": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm


##KMeans Clustering

def assign_risk_label(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    high_risk_cluster = rfm.groupby("cluster")["Monetary"].mean().idxmin()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm

##Merge Target Back

def merge_target(df, rfm):
    return df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )
