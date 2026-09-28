"""FastAPI service for customer churn prediction (Task 4)."""
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 1, "PhoneService": "No",
                "MultipleLines": "No phone service", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85, "TotalCharges": 29.85,
            }
        }
    }


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    try:
        for col, enc in encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid category value: {e}")

    df = df[feature_names]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "prediction": "Churn" if pred == 1 else "No Churn",
        "churn_probability": round(prob, 4),
    }
