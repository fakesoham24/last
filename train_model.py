"""Train term-deposit model and persist sklearn Pipeline to models/."""

from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from src.preprocess import build_model_pipeline


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column {TARGET_COL!r} in {DATA_PATH}")

    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df[FEATURE_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        if col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype(str).str.strip()

    # Drop rows with bad numerics if any
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_model_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    joblib.dump(pipe, MODEL_PATH)
    meta = {
        "n_features": len(FEATURE_COLUMNS),
        "positive_class": "yes",
        "label_encoding": {"no": 0, "yes": 1},
    }
    joblib.dump(meta, METADATA_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
