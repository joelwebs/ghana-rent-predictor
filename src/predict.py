from pathlib import Path

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rent_price_model.joblib"


# Features expected by the trained model
REQUIRED_FEATURES = [
    "listing_type",
    "category",
    "bedrooms",
    "bathrooms",
    "floor_area",
    "location",
    "condition",
    "region",
    "locality",
    "parking_space",
    "is_furnished",
    "lat",
    "lng",
    "amenities_count",
]


def load_model():
    """Load the trained machine learning pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_price(property_data):
    """
    Predict the listed price of a property.

    Args:
        property_data (dict): Property characteristics.

    Returns:
        float: Predicted listed price in GHS.
    """
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in property_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Arrange input in the expected feature order
    property_df = pd.DataFrame(
        [property_data]
    )[REQUIRED_FEATURES]

    model = load_model()

    prediction = model.predict(property_df)

    return float(prediction[0])
