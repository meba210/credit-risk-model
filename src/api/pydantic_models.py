from pydantic import BaseModel, Field
from typing import Optional


# -----------------------------
# INPUT SCHEMA (REQUEST)
# -----------------------------

class PredictionRequest(BaseModel):
    """
    Schema for incoming prediction requests.
    The fields must match the features used by the trained model.
    """

    Amount: float = Field(..., example=2500.0, description="Transaction amount")
    Value: float = Field(..., example=2500.0, description="Absolute transaction value")
    CountryCode: int = Field(..., example=256, description="Country code")
    PricingStrategy: int = Field(..., example=2, description="Pricing strategy category")

    transaction_hour: int = Field(
        ..., example=14, ge=0, le=23, description="Hour of transaction"
    )
    transaction_day: int = Field(
        ..., example=15, ge=1, le=31, description="Day of month"
    )
    transaction_month: int = Field(
        ..., example=6, ge=1, le=12, description="Month of year"
    )


# -----------------------------
# OUTPUT SCHEMA (RESPONSE)
# -----------------------------

class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """

    risk_probability: float = Field(
        ..., example=0.23, description="Predicted probability of high credit risk"
    )
    risk_label: int = Field(
        ..., example=0, description="0 = Low Risk, 1 = High Risk"
    )
