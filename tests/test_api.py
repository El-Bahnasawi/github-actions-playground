from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_neutral():
    resp = client.post("/predict", json={"text": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in {"neutral", "toxic"}
    assert 0.0 <= data["score"] <= 1.0


def test_predict_toxic_long_text():
    text = "x" * 200
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "toxic"
    assert 0.5 <= data["score"] <= 1.0
