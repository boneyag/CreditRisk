import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from creditrisk.api import app, PredictionRequest

valid_payload = {
    "age": 42,
    "occupation_status": "employed",
    "years_employed": 10,
    "annual_income": 85000.0,
    "credit_score": 720.0,
    "credit_history_years": 12,
    "savings_assets": 15000.0,
    "current_debt": 12000.0,
    "defaults_on_file": 0,
    "delinquencies_last_2yrs": 1,
    "derogatory_marks": 0,
    "product_type": "personal",
    "loan_intent": "debt_consolidation",
    "loan_amount": 15000.0,
    "interest_rate": 0.12,
    "debt_to_income_ratio": 0.18,
    "loan_to_income_ratio": 0.35,
    "payment_to_income_ratio": 0.08,
}

@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client

def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "xgb"
    assert body["health"] == "live"


def test_predict_endpoint(api_client):
    response = api_client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert "pred" in body
    assert "proba" in body
    assert "model_version" in body


def test_prediction_request_validation():
    request = PredictionRequest.model_validate(valid_payload)
    assert request.age == 42


def test_prediction_request_missing_field():
    bad_payload = valid_payload.copy()
    bad_payload.pop("age")
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(bad_payload)