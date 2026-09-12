from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/random_forest_model.pkl")
INPUT_DATA_PATH = Path("data/processed/X_features.csv")
OUTPUT_DATA_PATH = Path("artifacts/predictions.csv")


def load_model(model_path: Path = MODEL_PATH):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def load_data(data_path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Load prediction input data from a CSV file."""
    return pd.read_csv(data_path)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return the feature data unchanged."""
    return data


def make_predictions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions and class probabilities when available."""
    predictions = pd.DataFrame({"prediction": model.predict(X)})

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_labels = getattr(model, "classes_", range(probabilities.shape[1]))
        probability_columns = [
            f"probability_{class_label}" for class_label in class_labels
        ]
        probability_frame = pd.DataFrame(
            probabilities,
            columns=probability_columns,
            index=X.index,
        )
        predictions = pd.concat([predictions, probability_frame], axis=1)

    return predictions


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path = OUTPUT_DATA_PATH,
) -> None:
    """Save predictions to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def main() -> None:
    """Run the model prediction workflow."""
    model = load_model()
    data = load_data()
    X = prepare_features(data)
    predictions = make_predictions(model, X)
    save_predictions(predictions)

    print("First 10 predictions:")
    print(predictions.head(10))
    print(f"Total number of predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
