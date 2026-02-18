import pandas as pd
from src.data_processing import clean_data
from src.features import create_features
from src.model import train_model

def sample_data():
    return pd.DataFrame({
        "annual_income": [1000, 2000, 1500, 2500],
        "debt": [100, 200, 150, 250],
        "transaction_amount": [500, 600, 550, 650],
        "transaction_count": [5, 6, 5, 6],
        "is_high_risk": [0, 1, 0, 1]
    })

def test_clean_data():
    df = sample_data()
    df.loc[0, "annual_income"] = None
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0

def test_feature_creation():
    df = create_features(sample_data())
    assert "income_to_debt" in df.columns

def test_model_training():
    results = train_model(sample_data(), "is_high_risk")
    assert results["accuracy"] >= 0

def test_roc_auc_exists():
    results = train_model(sample_data(), "is_high_risk")
    assert "roc_auc" in results

def test_f1_exists():
    results = train_model(sample_data(), "is_high_risk")
    assert "f1" in results
