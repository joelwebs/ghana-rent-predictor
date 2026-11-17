
import sys
from pathlib import Path
import math

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_price


@pytest.fixture
def valid_property():
    """Sample property information for prediction testing."""
    return {
        "listing_type": "rentals",
        "category": "Flats",
        "bedrooms": 2,
        "bathrooms": 2,
        "floor_area": 100,
        "location": "Greater Accra, East Legon",
        "condition": "New",
        "region": "Greater Accra",
        "locality": "East Legon",
        "parking_space": 0,
        "is_furnished": "Unfurnished",
        "lat": 5.6358,
        "lng": -0.1613,
        "amenities_count": 5,
    }


def test_prediction_returns_number(valid_property):
    """A valid property should produce a numerical prediction."""
    prediction = predict_price(valid_property)

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)


def test_prediction_is_positive(valid_property):
    """The predicted listed price should be positive."""
    prediction = predict_price(valid_property)

    assert prediction > 0


def test_missing_features_raise_error(valid_property):
    """Missing required features should raise a ValueError."""
    valid_property.pop("bedrooms")

    with pytest.raises(ValueError, match="Missing required features"):
        predict_price(valid_property)
