import pandas as pd
from src.model import train_model
from src.features import create_features

def sample_data():
    return pd.DataFrame({
        "annual_income": [1000, 2000, 1500, 2500],
        "debt": [100, 200, 150, 250],
        "transaction_amount": [500, 600, 550, 650],
        "transaction_count": [5, 6, 5, 6],
        "is_high_risk": [0, 1, 0, 1]
    })

def test_train_model():
    df = sample_data()
    df = create_features(df)

    results = train_model(df, "is_high_risk")

    assert "model" in results
    assert results["accuracy"] >= 0
    assert "roc_auc" in results
    assert "f1" in results
