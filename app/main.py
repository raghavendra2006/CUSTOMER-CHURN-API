"""
FastAPI application for the Customer Churn Prediction API.

Exposes:
- GET  /         — Health check
- GET  /health   — Extended health check (confirms model is loaded)
- POST /predict  — Churn prediction endpoint
"""

import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.utils import load_pipeline, format_prediction_response
from app.model import ALL_FEATURES


# ---------------------------------------------------------------------------
# Pydantic Input Validation Model
# ---------------------------------------------------------------------------

class CustomerData(BaseModel):
    """
    Schema for customer data sent to the /predict endpoint.

    All 19 feature fields are required. Pydantic automatically validates
    types and returns HTTP 422 if any field is missing or has the wrong type.
    """
    gender: str = Field(..., description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., description="Senior citizen flag: 0 or 1")
    Partner: str = Field(..., description="Has partner: Yes or No")
    Dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure: int = Field(..., description="Number of months with company")
    PhoneService: str = Field(..., description="Has phone service: Yes or No")
    MultipleLines: str = Field(..., description="Multiple lines: Yes, No, or No phone service")
    InternetService: str = Field(..., description="Internet service type: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Online security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Online backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Device protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Tech support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Streaming movies: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., description="Monthly charge amount")
    TotalCharges: float = Field(..., description="Total charges to date")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "A RESTful API for predicting customer churn using a trained "
        "scikit-learn RandomForest pipeline. Send customer data to the "
        "/predict endpoint and receive a churn prediction with probability."
    ),
    version="1.0.0"
)

# Load the model pipeline once at startup — NOT on every request
pipeline = None


@app.on_event("startup")
def startup_event():
    """Load the ML pipeline into memory when the application starts."""
    global pipeline
    try:
        pipeline = load_pipeline()
        print("✅ Model pipeline loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("   The /predict endpoint will return 500 until the model is available.")


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a clean 500 response."""
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "Customer Churn Prediction API"}


@app.get("/health", tags=["Health"])
def health_check():
    """Extended health check — confirms the model is loaded and ready."""
    model_loaded = pipeline is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict", tags=["Prediction"])
def predict_churn(data: CustomerData):
    """
    Predict whether a customer will churn.

    Accepts customer feature data as JSON, validates all fields via Pydantic,
    runs the data through the trained scikit-learn pipeline, and returns
    the prediction with its probability.

    Returns:
        JSON with keys:
        - prediction: "Yes" or "No"
        - probability: float between 0 and 1 (probability of churn)
    """
    # Ensure model is loaded
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure the model file exists."
        )

    try:
        # Convert Pydantic model to dict, then to DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # Ensure column order matches training data
        df = df[ALL_FEATURES]

        # Run prediction
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0]

        # Get the probability of the positive class ('Yes' = churn)
        # Find the index of 'Yes' in the classifier's classes
        classes = pipeline.classes_.tolist()
        churn_index = classes.index('Yes') if 'Yes' in classes else 1
        churn_probability = probability[churn_index]

        return format_prediction_response(prediction, churn_probability)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
