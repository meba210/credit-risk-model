from fastapi import FastAPI
import pandas as pd
from utils.model_utils import load_model
from src.api.schemas import CustomerInput, PredictionOutput

app = FastAPI()

model = load_model("model.joblib")

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerInput):
    df = pd.DataFrame([data.dict()])
    probability = model.predict_proba(df)[0][1]
    return PredictionOutput(risk_probability=probability)
