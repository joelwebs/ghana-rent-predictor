import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_data


@pytest.fixture
def sample_data():
    """Create a small sample rental dataset."""

    return pd.DataFrame({
        "price": [3500, 5000, 3500, -100, None],
        "bedrooms": ["2", "3", "2", "1", "2"],
        "bathrooms": ["2", "3", "2", "1", "2"],
        "floor_area": ["120", "180", "120", "80", "100"],
        "lat": ["5.6037", "5.6500", "5.6037", "5.6000", "5.6100"],
        "lng": ["-0.1870", "-0.1700", "-0.1870", "-0.1800", "-0.1900"],
        "listing_type": ["rentals", "rentals", "rentals", "rentals", "rentals"],
        "category": ["Flats", " Duplex ", "Flats", "Flats", "Flats"],
        "location": ["East Legon", "Madina", "East Legon", "Accra", "Accra"],
        "condition": ["New", "Good", "New", "Good", "Good"],
        "region": ["Greater Accra"] * 5,
        "locality": ["Accra"] * 5,
        "is_furnished": ["Furnished"] * 5,
        "parking_space": [True, False, True, False, True],
    })


def test_clean_data_removes_duplicates(sample_data):
    """Duplicate rows should be removed."""

    cleaned = clean_data(sample_data)

    assert not cleaned.duplicated().any()


def test_clean_data_removes_invalid_prices(sample_data):
    """Missing and non-positive prices should be removed."""

    cleaned = clean_data(sample_data)

    assert cleaned["price"].notna().all()
    assert (cleaned["price"] > 0).all()


def test_clean_data_converts_numeric_columns(sample_data):
    """Numeric columns should have numeric data types."""

    cleaned = clean_data(sample_data)

    numeric_columns = [
        "price",
        "bedrooms",
        "bathrooms",
        "floor_area",
        "lat",
        "lng",
    ]

    for column in numeric_columns:
        assert pd.api.types.is_numeric_dtype(cleaned[column])


def test_clean_data_strips_categorical_whitespace(sample_data):
    """Extra whitespace should be removed from text values."""

    cleaned = clean_data(sample_data)

    assert "Duplex" in cleaned["category"].values


def test_clean_data_converts_parking_space(sample_data):
    """Parking values should be represented as 0 or 1."""

    cleaned = clean_data(sample_data)

    assert set(cleaned["parking_space"].dropna().unique()).issubset(
        {0, 1}
    )


def test_clean_data_does_not_modify_original(sample_data):
    """Cleaning should not modify the original DataFrame."""

    original = sample_data.copy(deep=True)

    clean_data(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)
