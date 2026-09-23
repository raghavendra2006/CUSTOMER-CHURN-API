"""
Training script for the Customer Churn Prediction model.

Run this script once to train the model and save it to disk:
    python train.py

The trained pipeline will be saved to models/churn_pipeline.joblib
"""

from app.model import load_data, train_and_save_model


def main():
    print("=" * 60)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 60)
    print()

    # Step 1: Load and clean data
    print("[1/3] Loading dataset...")
    df = load_data()
    print(f"      Dataset shape: {df.shape}")
    print(f"      Target distribution:")
    churn_counts = df['Churn'].value_counts()
    for label, count in churn_counts.items():
        pct = count / len(df) * 100
        print(f"        {label}: {count} ({pct:.1f}%)")
    print()

    # Step 2: Train and evaluate
    print("[2/3] Training model...")
    metrics = train_and_save_model(df)

    # Step 3: Summary
    print("[3/3] Training complete!")
    print()
    print("=" * 60)
    print("  Final Metrics Summary")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize():>10}: {value}")
    print("=" * 60)
    print()
    print("You can now start the API with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print()


if __name__ == '__main__':
    main()
