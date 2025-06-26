from pydantic import BaseModel

class RiskRequest(BaseModel):
    recency_days: float
    frequency: float
    monetary: float

class RiskResponse(BaseModel):
    risk_score: float 