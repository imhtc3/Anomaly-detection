"""Serve the trained detector.

Run:  python train.py  &&  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from detector.model import MIN_HISTORY, score_latest

app = FastAPI(title="Metric Anomaly Detector")
with open("model.pkl", "rb") as f:
    detector = pickle.load(f)


class MetricPoint(BaseModel):
    timestamp: str
    requests: float
    cpu_pct: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0)
    error_pct: float = Field(ge=0)
    memory_pct: float = Field(ge=0, le=100)


class ScoreRequest(BaseModel):
    points: list[MetricPoint] = Field(
        description=f"At least {MIN_HISTORY} per-minute points, oldest first; the last one is scored")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    df = pd.DataFrame([p.model_dump() for p in req.points])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    try:
        return score_latest(detector, df)
    except ValueError as e:
        raise HTTPException(400, str(e))
