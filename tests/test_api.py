"""
Integration tests for the Customer Churn Prediction API endpoints.

Tests cover:
- Health check endpoints
- Valid prediction requests
- Input validation (missing fields, wrong types, empty body)
- Response schema compliance
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_customer_data():
    """A complete, valid customer data payload."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_health_check(self, client):
        """GET / should return 200 with status healthy."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint(self, client):
        """GET /health should return 200 with model_loaded status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "status" in data


# ---------------------------------------------------------------------------
# Prediction Endpoint Tests — Valid Input
# ---------------------------------------------------------------------------

class TestPredictValidInput:
    """Tests for the /predict endpoint with valid data."""

    def test_predict_valid_input_returns_200(self, client, valid_customer_data):
        """POST /predict with valid data should return 200 OK."""
        response = client.post("/predict", json=valid_customer_data)
        assert response.status_code == 200

    def test_predict_response_has_prediction_key(self, client, valid_customer_data):
        """Response must contain the 'prediction' key."""
        response = client.post("/predict", json=valid_customer_data)
        data = response.json()
        assert "prediction" in data

    def test_predict_response_has_probability_key(self, client, valid_customer_data):
        """Response must contain the 'probability' key."""
        response = client.post("/predict", json=valid_customer_data)
        data = response.json()
        assert "probability" in data

    def test_predict_prediction_value(self, client, valid_customer_data):
        """Prediction must be either 'Yes' or 'No'."""
        response = client.post("/predict", json=valid_customer_data)
        data = response.json()
        assert data["prediction"] in ["Yes", "No"]

    def test_predict_probability_range(self, client, valid_customer_data):
        """Probability must be a float between 0 and 1."""
        response = client.post("/predict", json=valid_customer_data)
        data = response.json()
        prob = data["probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_response_schema(self, client, valid_customer_data):
        """Response should have exactly the keys 'prediction' and 'probability'."""
        response = client.post("/predict", json=valid_customer_data)
        data = response.json()
        assert set(data.keys()) == {"prediction", "probability"}


# ---------------------------------------------------------------------------
# Prediction Endpoint Tests — Invalid Input
# ---------------------------------------------------------------------------

class TestPredictInvalidInput:
    """Tests for the /predict endpoint with invalid data."""

    def test_predict_missing_field(self, client, valid_customer_data):
        """Omitting a required field should return 422."""
        # Remove 'tenure' field
        incomplete_data = valid_customer_data.copy()
        del incomplete_data["tenure"]
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_predict_wrong_type_integer(self, client, valid_customer_data):
        """Sending a string for an integer field should return 422."""
        bad_data = valid_customer_data.copy()
        bad_data["tenure"] = "not_a_number"
        response = client.post("/predict", json=bad_data)
        assert response.status_code == 422

    def test_predict_wrong_type_float(self, client, valid_customer_data):
        """Sending a string for a float field should return 422."""
        bad_data = valid_customer_data.copy()
        bad_data["MonthlyCharges"] = "expensive"
        response = client.post("/predict", json=bad_data)
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        """Sending an empty JSON body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_no_body(self, client):
        """Sending no body at all should return 422."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_multiple_missing_fields(self, client):
        """Sending only a few fields should return 422."""
        partial_data = {
            "gender": "Male",
            "tenure": 12
        }
        response = client.post("/predict", json=partial_data)
        assert response.status_code == 422
