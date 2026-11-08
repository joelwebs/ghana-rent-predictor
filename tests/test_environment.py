def test_ml_libraries_are_available():
    import numpy
    import pandas
    import sklearn

    assert numpy.__version__
    assert pandas.__version__
    assert sklearn.__version__
