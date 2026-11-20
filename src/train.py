"""Train and evaluate the Insurance Claim Fraud Detection model.

Run from the project root:
    python -m src.train

Optional arguments:
    --data data/raw/fraud_oracle.csv
    --capacity 200
    --test-size 0.20
    --random-state 42

This script mirrors the selected Random Forest configuration from Notebook 05,
while using the reusable data-processing and feature-engineering modules.
"""

from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import load_claims_csv, separate_target
from src.feature_engineering import engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_oracle.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model_pipeline.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "training_report.json"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create train-fitted numeric and categorical preprocessing pipelines."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build the selected balanced Random Forest model pipeline."""
    preprocessor = build_preprocessor(X_train)
    classifier = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    investigation_capacity: int = 200,
    total_claims_reference: int | None = None,
) -> dict[str, Any]:
    """Calculate threshold metrics and capacity-aligned Top-K metrics."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    # Scale the operational capacity to the held-out test-set size so that
    # evaluation is comparable to the full-population monthly capacity.
    if total_claims_reference is None:
        total_claims_reference = len(X_test)
    test_capacity = max(
        1,
        int(round(investigation_capacity * len(X_test) / total_claims_reference)),
    )
    test_capacity = min(test_capacity, len(X_test))

    ranking = np.argsort(probabilities)[::-1]
    top_k_indices = ranking[:test_capacity]
    fraud_in_top_k = int(y_test.iloc[top_k_indices].sum())

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "threshold": 0.50,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "investigation_capacity_full_population": int(investigation_capacity),
        "test_set_top_k": int(test_capacity),
        "fraud_captured_in_test_top_k": fraud_in_top_k,
        "precision_at_k": float(fraud_in_top_k / test_capacity),
        "recall_at_k": float(fraud_in_top_k / int(y_test.sum())),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_prevalence": float(y_test.mean()),
    }
    return metrics


def train_and_save(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    investigation_capacity: int = 200,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load, transform, split, train, evaluate, and persist model/report."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    report_path = Path(report_path)

    claims = load_claims_csv(data_path, validate_schema=True)
    X_raw, y = separate_target(claims, drop_identifiers=True)

    # Reusable feature transformations. The target is not included in X.
    X_engineered = engineer_features(claims, drop_identifiers=True, drop_target=True)

    # Guard against any accidental target/identifier leakage.
    forbidden_columns = {"FraudFound_P", "PolicyNumber", "RepNumber"}
    leaked = forbidden_columns.intersection(X_engineered.columns)
    if leaked:
        raise ValueError(f"Target/identifier leakage detected in features: {sorted(leaked)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        investigation_capacity=investigation_capacity,
        total_claims_reference=len(claims),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": "Random Forest (balanced_subsample)",
        "model_type": "RandomForestClassifier",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "original_predictor_count_after_id_drop": int(X_raw.shape[1]),
        "engineered_predictor_count": int(X_engineered.shape[1]),
        "target": "FraudFound_P",
        "feature_columns": X_engineered.columns.tolist(),
        "random_state": random_state,
        "test_size": test_size,
        "investigation_capacity": int(investigation_capacity),
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "metrics": metrics,
    }

    # Bundle model and metadata together to simplify later inference.
    joblib.dump({"pipeline": model, "metadata": metadata}, model_path)
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTRAINING COMPLETE")
    print("=" * 60)
    print(f"Data:                 {data_path}")
    print(f"Training rows:        {len(X_train):,}")
    print(f"Test rows:            {len(X_test):,}")
    print(f"Predictor features:   {X_engineered.shape[1]}")
    print(f"Model artifact:       {model_path}")
    print(f"Training report:       {report_path}")
    print("\nEvaluation metrics")
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            print(f"{metric_name}: {value}")
        elif isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the insurance fraud detection model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input claims CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output model artifact")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Output JSON report")
    parser.add_argument("--capacity", type=int, default=200, help="Investigation capacity for Top-K metrics")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Split random seed")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_path=args.data,
        model_path=args.model,
        report_path=args.report,
        investigation_capacity=args.capacity,
        test_size=args.test_size,
        random_state=args.random_state,
    )
