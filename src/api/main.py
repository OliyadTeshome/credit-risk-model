from fastapi import FastAPI
from .pydantic_models import RiskRequest, RiskResponse
from ..predict import load_model, predict_risk

app = FastAPI()
model = load_model()

@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk_endpoint(request: RiskRequest):
    risk_score = predict_risk(
        recency_days=request.recency_days,
        frequency=request.frequency,
        monetary=request.monetary,
        model=model
    )
    return RiskResponse(risk_score=round(risk_score, 4)) 