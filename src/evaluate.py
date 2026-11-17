import json
from pathlib import Path

import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import engineer_features, TARGET_COLUMN
from src.train import MODEL_PATH


BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_PATH = (
    BASE_DIR / "reports" / "metrics" / "model_evaluation.json"
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using regression metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
    }

    return metrics


def main():
    # Load and prepare dataset
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Recreate the same test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    model = joblib.load(MODEL_PATH)

    # Evaluate
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nModel Evaluation Results")
    print("------------------------")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Save metrics
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
