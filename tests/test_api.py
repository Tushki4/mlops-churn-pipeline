import os
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

import pytest
from fastapi.testclient import TestClient
from src.api import app

VALID_CUSTOMER = {
    "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict_valid_customer(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_tier" in data


def test_predict_probability_range(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_missing_field_returns_422(client):
    incomplete = {"tenure": 12, "MonthlyCharges": 65.5}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_risk_tier_is_valid(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    tier = response.json()["risk_tier"]
    assert tier in ["High Risk", "Medium Risk", "Low Risk"]