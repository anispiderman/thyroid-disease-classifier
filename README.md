# Thyroid Disease Classifier

A reproducible machine-learning pipeline for multiclass thyroid-disease classification using the UCI Thyroid Disease dataset. The project covers cleaning, missing-value handling, categorical encoding, feature construction, stratified splitting, model comparison, hyperparameter search, evaluation, and visual reporting.

## Models

- Logistic Regression with standardization
- K-Nearest Neighbours with standardization
- Random Forest

Model selection uses five-fold grid search on the training split. A fixed random seed and stratified 80/20 split support repeatability.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

Generated datasets and model artifacts are ignored. The raw data is attributed in `DATASET.md`, and evaluation figures are under `report/figures`.

## Project paper

I wrote a paper alongside my groupmate covering the dataset, preprocessing decisions, model comparison, results, and limitations.

## Repository layout

- `src/preprocessing_data`: parsing, cleaning, imputation, and encoding
- `src/feature_engineering`: feature/target construction
- `src/models`: model tuning, comparison, persistence, and prediction
- `src/visualization`: evaluation charts
- `notebooks`: exploratory analysis
- `report`: written report and figures

## Limitations

This is an educational classifier, not a medical device. It is not validated for clinical use and must not be used for diagnosis or treatment decisions. Several target classes are rare, so aggregate accuracy can conceal weak minority-class performance; per-class recall and F1 scores are essential.

## Collaboration

This classifier was built collaboratively as a student class project. We worked on the data pipeline, model training, evaluation, visualizations, and written report as a group.
