"""
Model training, evaluation, and serialization logic for Customer Churn Prediction.

This module handles:
- Loading and cleaning the Telco Customer Churn dataset
- Building a scikit-learn Pipeline with ColumnTransformer
- Training a RandomForestClassifier
- Evaluating and persisting the trained pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# Define feature columns used by the model
NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = 'Churn'


def load_data(filepath='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load the Telco Customer Churn CSV dataset and perform initial cleaning.

    Handles:
    - Blank strings in 'TotalCharges' (replaced with 0.0)
    - Drops 'customerID' as it is not a predictive feature

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for model training.
    """
    df = pd.read_csv(filepath)

    # TotalCharges has blank strings (' ') for new customers with tenure=0
    # Replace blank strings with '0' then convert to numeric
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill any remaining NaN values in TotalCharges with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Drop customerID — it's a unique identifier, not a feature
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    return df


def build_preprocessor():
    """
    Build a ColumnTransformer that applies:
    - StandardScaler to numerical features
    - OneHotEncoder to categorical features

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor


def train_and_save_model(df, model_path='models/churn_pipeline.joblib'):
    """
    Train a RandomForestClassifier pipeline and save it to disk.

    Steps:
    1. Separate features (X) and target (y)
    2. Train/test split (80/20, stratified)
    3. Build Pipeline (preprocessor + classifier)
    4. Fit on training data
    5. Evaluate on test data
    6. Persist pipeline to disk

    Args:
        df: Cleaned DataFrame from load_data().
        model_path: Path to save the serialized pipeline.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1).
    """
    # Separate features and target
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Train/test split — stratified to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the full pipeline
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    print("Training RandomForest classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, pos_label='Yes'), 4),
        'recall': round(recall_score(y_test, y_pred, pos_label='Yes'), 4),
        'f1_score': round(f1_score(y_test, y_pred, pos_label='Yes'), 4)
    }

    print("\n=== Model Evaluation Metrics ===")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1-Score:  {metrics['f1_score']}")
    print("================================\n")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Persist the full pipeline (preprocessor + classifier)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to: {model_path}")

    return metrics
