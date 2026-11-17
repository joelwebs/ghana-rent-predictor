import pandas as pd

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "floor_area",
    "parking_space",
    "lat",
    "lng",
    "amenities_count",
]

CATEGORICAL_FEATURES = [
    "listing_type",
    "category",
    "location",
    "condition",
    "region",
    "locality",
    "is_furnished",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "price"


def count_amenities(value):
    """
    Count the number of amenities in a listing.
    """
    if pd.isna(value) or str(value).strip() == "":
        return 0

    return len([
        item
        for item in str(value).split(",")
        if item.strip()
    ])


def engineer_features(df):
    """
    Create additional features for machine learning.

    Returns:
        DataFrame containing engineered features and target.
    """
    df = df.copy()

    if "amenities" not in df.columns:
        raise ValueError("Missing required column: amenities")

    df["amenities_count"] = (
        df["amenities"].apply(count_amenities)
    )

    # Keep only model features and target
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    engineered_df = df[required_columns].copy()

    print(
        f"Feature engineering completed: "
        f"{engineered_df.shape}"
    )

    return engineered_df
