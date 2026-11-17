import importlib

def test_ml_libraries_are_available():
    """Check that essential ML libraries are installed."""

    required_libraries = [
        "pandas",
        "numpy",
        "sklearn",
        "joblib",
        "pytest",
    ]

    for library in required_libraries:
        module = importlib.import_module(library)
        assert module is not None

