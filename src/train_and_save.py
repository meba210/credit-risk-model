import pandas as pd
from src.features import create_features
from src.data_processing import clean_data
from src.model import train_model
from utils.model_utils import save_model

def main():
    # Example training dataset (replace with your real processed dataset)
    df = pd.DataFrame({
        "annual_income": [1000, 2000, 1500, 2500],
        "debt": [100, 200, 150, 250],
        "transaction_amount": [500, 600, 550, 650],
        "transaction_count": [5, 6, 5, 6],
        "is_high_risk": [0, 1, 0, 1]
    })

    df = clean_data(df)
    df = create_features(df)

    results = train_model(df, "is_high_risk")

    save_model(results["model"], "model.joblib")

    print("✅ Model trained and saved as model.joblib")
    print("ROC-AUC:", results["roc_auc"])
    print("F1:", results["f1"])

if __name__ == "__main__":
    main()
