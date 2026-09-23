"""
Helper functions for the Customer Churn Prediction API.

Provides utilities for model loading and response formatting.
"""

import os
import joblib


# Default model path — can be overridden via MODEL_PATH env var
DEFAULT_MODEL_PATH = 'models/churn_pipeline.joblib'


def get_model_path():
    """
    Get the path to the serialized model pipeline.

    Checks the MODEL_PATH environment variable first, falls back to default.

    Returns:
        str: Absolute or relative path to the .joblib file.
    """
    return os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)


def load_pipeline(model_path=None):
    """
    Load a serialized scikit-learn pipeline from disk.

    Args:
        model_path: Optional path override. Uses get_model_path() if None.

    Returns:
        sklearn.pipeline.Pipeline: The loaded pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if model_path is None:
        model_path = get_model_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please run 'python train.py' to train and save the model first."
        )

    return joblib.load(model_path)


def format_prediction_response(prediction, probability):
    """
    Format the model's raw output into the expected API response schema.

    Args:
        prediction: Raw prediction from pipeline.predict() (e.g., 'Yes' or 'No').
        probability: Probability of churn from pipeline.predict_proba().

    Returns:
        dict: Response dict with keys 'prediction' and 'probability'.
    """
    return {
        "prediction": str(prediction),
        "probability": round(float(probability), 4)
    }
