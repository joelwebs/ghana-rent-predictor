from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "house_rentals.csv"

def load_data(file_path=RAW_DATA_PATH):
    """
    Load the Ghana house rental dataset.

    Args:
        file_path: Location of the CSV dataset.

    Returns:
        pandas DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print(f"Dataset loaded successfully: {df.shape}")

    return df
