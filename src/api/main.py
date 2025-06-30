from fastapi import FastAPI
from .pydantic_models import RiskRequest, RiskResponse
import joblib
import numpy as np
import os

app = FastAPI()

# Load the best model from the saved joblib file
MODEL_PATH = os.path.join('models', 'credit_risk_model.joblib')
model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=RiskResponse)
def predict_risk_endpoint(request: RiskRequest):
    features = np.array([[request.recency_days, request.frequency, request.monetary_total, request.monetary_avg, request.cluster]])
    risk_score = model.predict_proba(features)[0, 1]  # Get probability of positive class
    return RiskResponse(risk_score=float(risk_score)) 