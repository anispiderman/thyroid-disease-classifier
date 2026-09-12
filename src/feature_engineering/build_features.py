from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/thyroid0387_preprocessed.csv")
X_OUTPUT_PATH = Path("data/processed/X_features.csv")
Y_OUTPUT_PATH = Path("data/processed/y_target.csv")
TARGET_COLUMN = "diagnosis_primary"
DROP_COLUMNS = [TARGET_COLUMN, "diagnosis", "record_id"]


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    return pd.read_csv(data_path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature data (X) and target data (y)."""
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN]
    return X, y


def save_data(
    X: pd.DataFrame,
    y: pd.Series,
    x_output_path: Path = X_OUTPUT_PATH,
    y_output_path: Path = Y_OUTPUT_PATH,
) -> None:
    """Save the feature matrix and target column to CSV files."""
    x_output_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(x_output_path, index=False)
    y.to_frame(name=TARGET_COLUMN).to_csv(y_output_path, index=False)


def main() -> None:
    """Run the feature building pipeline."""
    df = load_data()
    X, y = build_features(df)
    save_data(X, y)

    print(f"Saved features to {X_OUTPUT_PATH}")
    print(f"Saved target to {Y_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
