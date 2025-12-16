from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()

model = mlflow.sklearn.load_model("models:/credit-risk-model/Production")

@app.post("/predict")
def predict(data: dict):
    prob = model.predict_proba(pd.DataFrame([data]))[:, 1]
    return {"risk_probability": float(prob)}
