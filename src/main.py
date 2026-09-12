from importlib import util
from pathlib import Path

from feature_engineering.build_features import main as build_features
from models.predict_model import main as predict_model
from models.train_model import main as train_model
from visualization.visualize import load_data, run_visualizations 

PREPROCESSING_SCRIPT_PATH = (
    Path(__file__).resolve().parent / "preprocessing_data" / "pre-processing.py"
)


def load_preprocess_function():
    """Load preprocess_thyroid_data from the preprocessing script."""
    spec = util.spec_from_file_location(
        "src.preprocessing_data.pre_processing",
        PREPROCESSING_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load preprocessing script from {PREPROCESSING_SCRIPT_PATH}"
        )

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_thyroid_data


preprocess_thyroid_data = load_preprocess_function()


def main() -> None:
    """Run the pipeline excluding visualization."""
    preprocess_thyroid_data()
    print("Preprocessing complete")

    build_features()
    print("Feature engineering complete")

    train_model()
    print("Model training complete")

    predict_model()
    print("Prediction complete")

    y_true, y_pred = load_data() 
    run_visualizations(y_true, y_pred)
    print("Visualizations complete")


if __name__ == "__main__":
    main()
