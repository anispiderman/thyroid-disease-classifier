from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


TRUE_LABELS_PATH = Path("data/processed/y_target.csv")
PREDICTIONS_PATH = Path("artifacts/predictions.csv")
FIGURES_DIR = Path("report/figures")
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
TRUE_DISTRIBUTION_PATH = FIGURES_DIR / "true_class_distribution.png"
PREDICTED_DISTRIBUTION_PATH = FIGURES_DIR / "predicted_class_distribution.png"
COMPARISON_DISTRIBUTION_PATH = FIGURES_DIR / "true_vs_predicted_distribution.png"
CLASS_METRICS_PATH = FIGURES_DIR / "per_class_precision_recall_f1.png"


def load_data(
    true_labels_path: Path = TRUE_LABELS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> tuple[pd.Series, pd.Series]:
    """Load true labels and predicted labels from CSV files."""
    y_true = pd.read_csv(true_labels_path).squeeze("columns")
    predictions = pd.read_csv(predictions_path)
    y_pred = predictions["prediction"]
    return y_true, y_pred


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Plot and save a normalized confusion matrix heatmap."""
    labels = sorted(pd.Index(y_true).union(pd.Index(y_pred)).unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_distribution(
    values: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Plot and save a class distribution percentage bar chart."""
    class_percentages = values.value_counts(normalize=True).mul(100).sort_values(
        ascending=False
    )
    colors = ["red" if label == "-" else "steelblue" for label in class_percentages.index]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        x=class_percentages.index,
        y=class_percentages.values,
        ax=ax,
        palette=colors,
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_distribution_comparison(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path = COMPARISON_DISTRIBUTION_PATH,
) -> None:
    """Plot and save the true vs predicted class distribution comparison."""
    true_distribution = y_true.value_counts(normalize=True)
    predicted_distribution = y_pred.value_counts(normalize=True)
    labels = sorted(set(true_distribution.index).union(predicted_distribution.index))

    distribution_df = pd.DataFrame(
        {
            "True": true_distribution.reindex(labels, fill_value=0).mul(100),
            "Predicted": predicted_distribution.reindex(labels, fill_value=0).mul(100),
        }
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    distribution_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("True vs Predicted Class Distribution", fontsize=16)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(["True", "Predicted"])
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path = CLASS_METRICS_PATH,
) -> None:
    """Plot and save per-class precision, recall, and F1 scores."""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(
        index=["accuracy", "macro avg", "weighted avg"],
        errors="ignore",
    )[["precision", "recall", "f1-score"]]

    fig, ax = plt.subplots(figsize=(16, 8))
    metrics_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Per-Class Precision, Recall, F1 Score", fontsize=16)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_visualizations(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Generate all evaluation visualizations."""
    sns.set_theme(style="whitegrid")
    plot_confusion_matrix(y_true, y_pred)
    plot_distribution(y_true, "Class Distribution (Percentage)", TRUE_DISTRIBUTION_PATH)
    plot_distribution(y_pred, "Predicted Class Distribution", PREDICTED_DISTRIBUTION_PATH)
    plot_distribution_comparison(y_true, y_pred)
    plot_classification_metrics(y_true, y_pred)


def main() -> None:
    """Generate and save visualization figures."""
    y_true, y_pred = load_data()

    if len(y_true) != len(y_pred):
        raise ValueError(
            "The number of true labels does not match the number of predictions."
        )

    run_visualizations(y_true, y_pred)


if __name__ == "__main__":
    main()
