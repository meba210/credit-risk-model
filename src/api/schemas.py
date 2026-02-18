from pydantic import BaseModel

class CustomerInput(BaseModel):
    annual_income: float
    debt: float
    transaction_amount: float
    transaction_count: int

class PredictionOutput(BaseModel):
    risk_probability: float
