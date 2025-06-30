from pydantic import BaseModel

class RiskRequest(BaseModel):
    recency_days: float
    frequency: float
    monetary_total: float
    monetary_avg: float
    cluster: int

class RiskResponse(BaseModel):
    risk_score: float 