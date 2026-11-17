from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import (
    engineer_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "rent_price_model.joblib"


def build_pipeline():
    """
    Build the complete preprocessing and Random Forest pipeline.
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def train_model(X_train, y_train):
    """
    Train the Random Forest pipeline.
    """

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline


def main():
    # Load dataset
    df = load_data()

    # Clean dataset
    df = clean_data(df)

    # Engineer features
    df = engineer_features(df)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    model = train_model(X_train, y_train)

    # Create models directory
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
