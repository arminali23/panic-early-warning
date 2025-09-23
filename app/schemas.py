# app/schemas.py
from pydantic import BaseModel

class ScoreResponse(BaseModel):
    user_id: str
    risk: float
    details: dict
