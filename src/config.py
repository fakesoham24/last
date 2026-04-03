"""Paths and feature definitions shared by training and inference."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "term_deposit_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.joblib"

TARGET_COL = "y"

# Feature columns as in the raw dataset (without target)
FEATURE_COLUMNS: list[str] = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

NUMERIC_FEATURES: list[str] = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]

# Expected pandas dtypes after successful coercion (for validation messaging)
EXPECTED_DTYPES: dict[str, str] = {**{c: "numeric" for c in NUMERIC_FEATURES}, **{c: "categorical" for c in CATEGORICAL_FEATURES}}
