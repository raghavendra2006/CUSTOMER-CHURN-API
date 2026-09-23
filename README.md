# 🔮 Customer Churn Prediction API

A production-ready, end-to-end machine learning pipeline for predicting customer churn in the telecommunications industry. This project bridges applied data science with MLOps best practices — from data preprocessing and model training to RESTful API deployment and Docker containerization.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Getting Started](#getting-started)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Dataset](#dataset)

---

## Overview

Customer churn (attrition) is the rate at which customers stop doing business with a company. By accurately predicting which customers are at high risk of churning, businesses can proactively intervene with targeted retention strategies.

This project provides:
- **Data Preprocessing Pipeline** — Automated cleaning, encoding, and scaling via scikit-learn `ColumnTransformer`
- **Trained ML Model** — `RandomForestClassifier` serialized as a single `.joblib` pipeline
- **RESTful API** — FastAPI server with Pydantic validation, health checks, and structured error responses
- **Containerization** — Docker + Docker Compose for one-command deployment
- **Automated Tests** — 25 test cases covering API endpoints and model logic

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Environment                      │
│                                                              │
│   API Client (Postman / curl / Web App)                      │
│         │                          ▲                         │
│         │ POST /predict            │ JSON Response            │
│         │ (JSON Data)              │ {prediction, probability}│
│         ▼                          │                         │
├─────────────────────────────────────────────────────────────┤
│              Docker Container (Port 8000)                    │
│                                                              │
│   ┌──────────────────────────────────────┐                   │
│   │       FastAPI Server (main.py)       │                   │
│   │                                      │                   │
│   │  ┌──────────────────────────────┐    │                   │
│   │  │  Pydantic Input Validation   │    │                   │
│   │  │  (CustomerData BaseModel)    │    │                   │
│   │  └──────────┬───────────────────┘    │                   │
│   │             │ Validated Features     │                   │
│   │             ▼                        │                   │
│   │  ┌──────────────────────────────┐    │                   │
│   │  │  Scikit-Learn Pipeline       │    │                   │
│   │  │  (churn_pipeline.joblib)     │    │                   │
│   │  │                              │    │                   │
│   │  │  ┌────────────────────────┐  │    │                   │
│   │  │  │ ColumnTransformer      │  │    │                   │
│   │  │  │ • StandardScaler (num) │  │    │                   │
│   │  │  │ • OneHotEncoder (cat)  │  │    │                   │
│   │  │  └──────────┬─────────────┘  │    │                   │
│   │  │             ▼                │    │                   │
│   │  │  ┌────────────────────────┐  │    │                   │
│   │  │  │ RandomForestClassifier │  │    │                   │
│   │  │  │ (100 estimators)       │  │    │                   │
│   │  │  └────────────────────────┘  │    │                   │
│   │  └──────────────────────────────┘    │                   │
│   └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **Python 3.9+** | Core language |
| **Pandas** | Data loading, cleaning, and manipulation |
| **NumPy** | Numerical computing |
| **scikit-learn** | ML pipeline — `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `RandomForestClassifier` |
| **FastAPI** | REST API framework with automatic OpenAPI docs |
| **Pydantic** | Request/response validation |
| **Uvicorn** | ASGI server |
| **Joblib** | Model serialization |
| **Docker** | Containerization |
| **Pytest** | Testing framework |

---

## Project Structure

```
CUSTOMER-CHURN-API/
├── app/
│   ├── __init__.py              # App package init
│   ├── main.py                  # FastAPI app, routes, Pydantic models
│   ├── model.py                 # ML pipeline: data loading, training, evaluation
│   └── utils.py                 # Helper functions: model loading, response formatting
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 rows)
├── models/
│   └── churn_pipeline.joblib    # Serialized sklearn Pipeline (preprocessor + classifier)
├── tests/
│   ├── __init__.py              # Tests package init
│   ├── test_api.py              # 12 API integration tests
│   └── test_model.py            # 13 model unit tests
├── train.py                     # Standalone training script
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Single-command orchestration
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Model Details

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles non-linear relationships between features without extensive tuning
- Robust against unscaled features (though we scale anyway for pipeline consistency)
- Provides feature importance rankings
- Less prone to overfitting than individual decision trees

### Preprocessing Pipeline

The model uses a unified `sklearn.pipeline.Pipeline` with a `ColumnTransformer`:

| Feature Type | Columns | Transformer |
|---|---|---|
| **Numerical** | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` |
| **Categorical** | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `OneHotEncoder(handle_unknown='ignore')` |

### Evaluation Metrics (Test Set — 20% holdout, stratified)

| Metric | Score |
|---|---|
| **Accuracy** | 0.7864 |
| **Precision** | 0.6254 |
| **Recall** | 0.4866 |
| **F1-Score** | 0.5474 |

> **Note:** For churn prediction, **Recall** is critical — capturing as many at-risk customers as possible — while **Precision** ensures retention resources aren't wasted on loyal customers. Metrics are reproducible with `random_state=42`.

### Data Cleaning

- **`TotalCharges`**: Contains blank strings (`' '`) for 11 new customers with `tenure=0`. Converted to `0.0`.
- **`customerID`**: Dropped — unique identifier with no predictive value.

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Docker)
- Git

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/raghavendra2006/CUSTOMER-CHURN-API.git
cd CUSTOMER-CHURN-API
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Configure environment variables**
```bash
cp .env.example .env
```

**4. Train the model**
```bash
python train.py
```

This will:
- Load and clean the Telco Customer Churn dataset
- Train a RandomForest pipeline with ColumnTransformer
- Print evaluation metrics (Accuracy, Precision, Recall, F1)
- Save the pipeline to `models/churn_pipeline.joblib`

**5. Start the API server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**6. Explore the interactive docs**

Open `http://localhost:8000/docs` for the Swagger UI.

---

### Docker Deployment

**Build and run with a single command:**
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

**Stop the container:**
```bash
docker-compose down
```

> **Important:** Train the model locally (`python train.py`) before building the Docker image. The Dockerfile copies the pre-trained `models/churn_pipeline.joblib` into the container.

---

## API Documentation

### Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `GET` | `/` | Health check | `200 OK` |
| `GET` | `/health` | Extended health check (model status) | `200 OK` |
| `POST` | `/predict` | Predict customer churn | `200 OK` |

### `POST /predict`

**Request Body** — All 19 fields are required:

```json
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
```

**Success Response** (`200 OK`):

```json
{
  "prediction": "Yes",
  "probability": 0.85
}
```

**Error Responses:**

| Status Code | Scenario |
|---|---|
| `422 Unprocessable Entity` | Missing required field or wrong data type |
| `500 Internal Server Error` | Model not loaded or unexpected server error |

### Example `curl` Commands

**Predict churn (valid request):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65
  }'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Invalid request (missing fields):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "tenure": 12}'
```
→ Returns `422 Unprocessable Entity`

---

## Testing

### Run the full test suite

```bash
pytest tests/ -v
```

### Test Categories

| File | Tests | Coverage |
|---|---|---|
| `tests/test_api.py` | 12 | Health checks, valid predictions, input validation (missing fields, wrong types, empty body) |
| `tests/test_model.py` | 13 | Data loading/cleaning, pipeline existence, predict/predict_proba output |
| **Total** | **25** | API + ML pipeline |

### Run specific test files

```bash
# API tests only
pytest tests/test_api.py -v

# Model tests only
pytest tests/test_model.py -v
```

---

## Dataset

**Telco Customer Churn** — IBM Sample Dataset

- **Source**: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Rows**: 7,043 customers
- **Columns**: 21 (19 features + customerID + Churn target)
- **Target**: `Churn` — binary (Yes/No)
- **Class Distribution**: ~73.5% No, ~26.5% Yes (imbalanced)

### Feature Descriptions

| Feature | Type | Values |
|---|---|---|
| `gender` | Categorical | Male, Female |
| `SeniorCitizen` | Numeric | 0, 1 |
| `Partner` | Categorical | Yes, No |
| `Dependents` | Categorical | Yes, No |
| `tenure` | Numeric | 0–72 months |
| `PhoneService` | Categorical | Yes, No |
| `MultipleLines` | Categorical | Yes, No, No phone service |
| `InternetService` | Categorical | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Yes, No, No internet service |
| `TechSupport` | Categorical | Yes, No, No internet service |
| `StreamingTV` | Categorical | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Yes, No, No internet service |
| `Contract` | Categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Yes, No |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | Numeric | 18.25–118.75 |
| `TotalCharges` | Numeric | 0–8684.80 |

---

## License

This project is for educational and portfolio purposes.