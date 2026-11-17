
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_price


@pytest.fixture
def valid_property():
    return {
        "listing_type": "rentals",
        "category": "Flats",
        "bedrooms": 2,
        "bathrooms": 2,
        "floor_area": 120,
        "location": "East Legon",
        "condition": "New",
        "region": "Greater Accra",
        "locality": "Accra",
        "parking_space": 1,
        "is_furnished": "Furnished",
        "lat": 5.6037,
        "lng": -0.1870,
        "amenities_count": 5,
    }


@pytest.fixture
def mock_model(monkeypatch):
    model = MagicMock()
    model.predict.return_value = [3500.0]

    monkeypatch.setattr(
        "src.predict.load_model",
        lambda: model
    )

    return model


def test_prediction_returns_number(valid_property, mock_model):
    prediction = predict_price(valid_property)

    assert isinstance(prediction, float)


def test_prediction_is_positive(valid_property, mock_model):
    prediction = predict_price(valid_property)

    assert prediction > 0


def test_prediction_uses_all_features(valid_property, mock_model):
    predict_price(valid_property)

    mock_model.predict.assert_called_once()

    input_data = mock_model.predict.call_args[0][0]

    assert list(input_data.columns) == list(valid_property.keys())


def test_missing_features_raise_error(valid_property):
    valid_property.pop("bedrooms")

    with pytest.raises(ValueError, match="bedrooms"):
        predict_price(valid_property)
