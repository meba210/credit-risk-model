import pytest
import pandas as pd
from src.features import create_features
from src.model import train_model
from src.data_processing import clean_data

def test_clean_data():
    df = pd.DataFrame({'a':[1,1,None]})
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0

def test_create_features():
    df = pd.DataFrame({'annual_income':[1000,2000],'debt':[100,200], 'transaction_amount':[500,600], 'transaction_count':[5,6]})
    df_feat = create_features(df)
    assert 'income_to_debt' in df_feat.columns
    assert 'avg_transaction' in df_feat.columns

def test_train_model():
    df = pd.DataFrame({
        'annual_income':[1000,2000,1500,2500],
        'debt':[100,200,150,250],
        'transaction_amount':[500,600,550,650],
        'transaction_count':[5,6,5,6],
        'is_high_risk':[0,1,0,1]
    })
    model, auc, f1 = train_model(df, 'is_high_risk')
    assert auc >= 0.5
    assert f1 >= 0
