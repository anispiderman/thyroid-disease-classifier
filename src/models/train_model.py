from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X_DATA_PATH = Path("data/processed/X_features.csv")
Y_DATA_PATH = Path("data/processed/y_target.csv")
MODEL_OUTPUT_PATH = Path("artifacts/random_forest_model.pkl")


def load_data(
    x_data_path: Path = X_DATA_PATH,
    y_data_path: Path = Y_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature data and target data from CSV files."""
    X = pd.read_csv(x_data_path)
    y = pd.read_csv(y_data_path).squeeze("columns")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Tune and train all models using GridSearchCV."""
    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        ]
    )
    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        cv=5,
        n_jobs=-1,
    )
    logistic_search.fit(X_train, y_train)

    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
    knn_search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid={"model__n_neighbors": [3, 5, 7, 9, 11, 13, 15]},
        cv=5,
        n_jobs=-1,
    )
    knn_search.fit(X_train, y_train)

    random_forest_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        },
        cv=5,
        n_jobs=-1,
    )
    random_forest_search.fit(X_train, y_train)

    return {
        "Logistic Regression": logistic_search,
        "K-Nearest Neighbors": knn_search,
        "Random Forest": random_forest_search,
    }


def evaluate_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print the best parameters and classification report for each model."""
    for model_name, search in trained_models.items():
        y_pred = search.predict(X_test)

        print(f"\n{model_name}")
        print(f"Best parameters: {search.best_params_}")
        print(classification_report(y_test, y_pred))


def save_model(model, output_path: Path = MODEL_OUTPUT_PATH) -> None:
    """Save the trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def main() -> None:
    """Run the model training, tuning, evaluation, and saving workflow."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_models = train_models(X_train, y_train)
    evaluate_models(trained_models, X_test, y_test)
    save_model(trained_models["Random Forest"].best_estimator_)

    print(f"\nSaved Random Forest model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
