from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_calculator_query():
    response = client.post("/query", json={"query": "What is 125 * (8 + 4)?"})
    assert response.status_code == 200
    data = response.json()
    assert "1500" in data["answer"]
    assert data["route"] == "calculator"
    assert len(data["trace"]) >= 3

def test_empty_query_rejected():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
