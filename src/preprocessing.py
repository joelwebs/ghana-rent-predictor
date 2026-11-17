import pandas as pd

NUMERIC_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "floor_area",
    "lat",
    "lng",
]

CATEGORICAL_COLUMNS = [
    "listing_type",
    "category",
    "location",
    "condition",
    "region",
    "locality",
    "is_furnished",
]

def clean_data(df):
    """
    Clean the raw rental dataset.

    Returns:
        Cleaned pandas DataFrame.
    """
    df = df.copy()

    # Remove exact duplicate records
    df = df.drop_duplicates()

    # Convert numerical columns
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Standardize categorical values
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # Convert parking_space to numeric 0/1
    if "parking_space" in df.columns:
        parking_mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
        }

        df["parking_space"] = (
            df["parking_space"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(parking_mapping)
        )

    # Remove records with missing or invalid target prices
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    df = df.reset_index(drop=True)

    print(f"Cleaning completed: {df.shape}")

    return df
