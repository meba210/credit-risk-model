from src.data_processing import calculate_rfm
import pandas as pd

def test_rfm_columns():
    df = pd.DataFrame({
        "CustomerId": [1, 1],
        "TransactionId": [10, 11],
        "Value": [100, 200],
        "TransactionStartTime": ["2024-01-01", "2024-01-05"]
    })

    rfm = calculate_rfm(df)
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
