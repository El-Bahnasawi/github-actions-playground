from fastapi import FastAPI
from pydantic import BaseModel

from ml_model import predict_label, score_text

app = FastAPI(title="GitHub Actions MLOps Playground")


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    text = payload.text
    score = score_text(text)
    label = predict_label(text)
    return PredictionOut(label=label, score=score)
