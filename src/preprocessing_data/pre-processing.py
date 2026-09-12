
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/thyroid0387.data")
DEFAULT_OUTPUT_PATH = Path("data/processed/thyroid0387_preprocessed.csv")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "i131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "tsh_measured",
    "tsh",
    "t3_measured",
    "t3",
    "tt4_measured",
    "tt4",
    "t4u_measured",
    "t4u",
    "fti_measured",
    "fti",
    "tbg_measured",
    "tbg",
    "referral_source",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["diagnosis_record"]
BOOLEAN_COLUMNS = [
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "i131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "tsh_measured",
    "t3_measured",
    "tt4_measured",
    "t4u_measured",
    "fti_measured",
    "tbg_measured",
]
NUMERIC_COLUMNS = ["age", "tsh", "t3", "tt4", "t4u", "fti", "tbg"]


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the thyroid data using the documented schema from thyroid0387.names."""
    return pd.read_csv(
        data_path,
        header=None,
        names=ALL_COLUMNS,
        na_values="?",
        keep_default_na=False,
    )


def split_diagnosis_and_record_id(df: pd.DataFrame) -> pd.DataFrame:
    """Separate the diagnosis label from the trailing record id."""
    diagnosis_parts = df["diagnosis_record"].astype(str).str.extract(
        r"^(?P<diagnosis>[^\[]+)\[(?P<record_id>\d+)\]$"
    )

    df["diagnosis"] = diagnosis_parts["diagnosis"].fillna(df["diagnosis_record"]).str.strip()
    df["record_id"] = diagnosis_parts["record_id"]

    # When the diagnosis is written as X|Y, the names file says Y is more likely.
    df["diagnosis_primary"] = df["diagnosis"].str.split("|", regex=False).str[-1]
    df = df.drop(columns=["diagnosis_record"])
    return df


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing decisions supported by the EDA."""
    df = split_diagnosis_and_record_id(df.copy())

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # EDA showed TBG is missing for 8823 of 9172 rows, so we drop the sparse lab value.
    df = df.drop(columns=["tbg"])

    # Keep missing sex explicit instead of forcing it into one of the observed categories.
    df["sex"] = df["sex"].replace("", pd.NA).fillna("Unknown")

    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].map({"t": 1, "f": 0}).astype("Int64")

    numeric_columns_to_impute = ["age", "tsh", "t3", "tt4", "t4u", "fti"]
    for column in numeric_columns_to_impute:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical gaps before encoding.
    df["referral_source"] = df["referral_source"].replace("", pd.NA).fillna("other")

    encoded = pd.get_dummies(
        df,
        columns=["sex", "referral_source"],
        drop_first=False,
        dtype=int,
    )

    return encoded


def preprocess_thyroid_data(
    data_path: Path = RAW_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the processed dataset."""
    df = load_raw_data(data_path)
    processed_df = clean_and_encode(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    return processed_df


if __name__ == "__main__":
    processed_df = preprocess_thyroid_data()
    print(f"Saved processed data to {DEFAULT_OUTPUT_PATH}")
    print(f"Processed shape: {processed_df.shape}")
