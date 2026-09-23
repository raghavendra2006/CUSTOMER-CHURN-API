"""
Unit tests for the model training and prediction logic.

Tests cover:
- Data loading and cleaning
- Pipeline file existence
- Pipeline prediction output format
"""

import os
import pytest
import pandas as pd
import numpy as np
from app.model import load_data, ALL_FEATURES, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from app.utils import load_pipeline


# ---------------------------------------------------------------------------
# Data Loading Tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests for the data loading and cleaning functions."""

    def test_load_data_returns_dataframe(self):
        """load_data() should return a pandas DataFrame."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_shape(self):
        """Dataset should have 7043 rows and 20 columns (after dropping customerID)."""
        df = load_data()
        assert df.shape[0] == 7043
        assert df.shape[1] == 20

    def test_load_data_no_customer_id(self):
        """customerID column should be dropped."""
        df = load_data()
        assert 'customerID' not in df.columns

    def test_load_data_total_charges_numeric(self):
        """TotalCharges should be numeric with no NaN values."""
        df = load_data()
        assert df['TotalCharges'].dtype in [np.float64, np.float32]
        assert df['TotalCharges'].isna().sum() == 0

    def test_load_data_has_target_column(self):
        """Dataset must contain the Churn target column."""
        df = load_data()
        assert TARGET_COLUMN in df.columns

    def test_load_data_has_all_features(self):
        """Dataset must contain all expected feature columns."""
        df = load_data()
        for feature in ALL_FEATURES:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_load_data_target_values(self):
        """Target column should only contain 'Yes' and 'No'."""
        df = load_data()
        unique_values = set(df[TARGET_COLUMN].unique())
        assert unique_values == {'Yes', 'No'}


# ---------------------------------------------------------------------------
# Model Pipeline Tests
# ---------------------------------------------------------------------------

class TestModelPipeline:
    """Tests for the saved model pipeline."""

    def test_pipeline_file_exists(self):
        """The serialized pipeline file should exist in models/."""
        assert os.path.exists('models/churn_pipeline.joblib'), \
            "Model file not found. Run 'python train.py' first."

    def test_pipeline_loads_successfully(self):
        """The pipeline should load without errors."""
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_pipeline_has_predict_method(self):
        """Loaded pipeline must have a predict method."""
        pipeline = load_pipeline()
        assert hasattr(pipeline, 'predict')

    def test_pipeline_has_predict_proba_method(self):
        """Loaded pipeline must have a predict_proba method."""
        pipeline = load_pipeline()
        assert hasattr(pipeline, 'predict_proba')

    def test_pipeline_predict_single_sample(self):
        """Pipeline should return a prediction for a single sample."""
        pipeline = load_pipeline()

        # Create a sample input matching training features
        sample = pd.DataFrame([{
            'SeniorCitizen': 0,
            'tenure': 1,
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85,
            'gender': 'Female',
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }])

        # Reorder columns to match ALL_FEATURES
        sample = sample[ALL_FEATURES]

        prediction = pipeline.predict(sample)
        assert len(prediction) == 1
        assert prediction[0] in ['Yes', 'No']

    def test_pipeline_predict_proba_shape(self):
        """predict_proba should return probabilities for both classes."""
        pipeline = load_pipeline()

        sample = pd.DataFrame([{
            'SeniorCitizen': 0,
            'tenure': 1,
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85,
            'gender': 'Female',
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }])

        sample = sample[ALL_FEATURES]

        proba = pipeline.predict_proba(sample)
        assert proba.shape == (1, 2)  # 2 classes: No, Yes
        assert abs(proba[0].sum() - 1.0) < 1e-6  # Probabilities sum to 1
