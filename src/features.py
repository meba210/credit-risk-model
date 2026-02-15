from typing import Any
import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for credit risk model.
    """
    # Example numerical features
    df['income_to_debt'] = df['annual_income'] / (df['debt'] + 1)
    df['avg_transaction'] = df['transaction_amount'] / (df['transaction_count'] + 1)

    # Example categorical encoding
    if 'product_category' in df.columns:
        df = pd.get_dummies(df, columns=['product_category'], drop_first=True)

    return df
