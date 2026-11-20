"""Score incoming insurance claims using the trained fraud model.

Run from the repository root:
    python -m src.predict --input data/raw/fraud_oracle.csv --output reports/scored_claims.csv

The input may be a CSV or a pandas DataFrame when using the Python API. The model
artifact is loaded from models/fraud_model_pipeline.joblib by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd

from src.data_processing import EXPECTED_COLUMNS, DataValidationError
from src.feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model_pipeline.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_oracle.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "scored_claims.csv"
DEFAULT_THRESHOLD = 0.50

# The target is not required for inference; identifier columns may be present in
# input but are excluded from the model predictors by feature engineering.
REQUIRED_INFERENCE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != "FraudFound_P"]


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[Any, dict]:
    """Load a bundled sklearn pipeline and its training metadata."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.train` first."
        )
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        metadata = artifact.get("metadata", {})
    else:
        # Compatibility with a model saved as a bare sklearn estimator/pipeline.
        pipeline = artifact
        metadata = {}
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")
    return pipeline, metadata


def validate_inference_data(claims: pd.DataFrame) -> None:
    """Validate that incoming claims contain the raw fields needed by the pipeline."""
    if not isinstance(claims, pd.DataFrame):
        raise TypeError("Input claims must be a pandas DataFrame.")
    if claims.empty:
        raise DataValidationError("Input claims dataset is empty.")
    if claims.columns.duplicated().any():
        duplicates = claims.columns[claims.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate column names detected: {duplicates}")
    missing = [c for c in REQUIRED_INFERENCE_COLUMNS if c not in claims.columns]
    if missing:
        raise DataValidationError(f"Missing required inference columns: {missing}")


def score_claims(
    claims: pd.DataFrame,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    include_features: bool = False,
) -> pd.DataFrame:
    """Score claim rows and return probabilities, binary predictions and risk bands.

    Parameters
    ----------
    claims:
        Raw, unlabeled claims. It may also contain FraudFound_P; this target is
        ignored for scoring and is not returned unless include_features=True.
    model_path:
        Path to the serialized model bundle.
    threshold:
        Probability threshold used for the binary predicted-fraud label.
    include_features:
        If True, return engineered predictors in addition to claim fields and
        prediction outputs. Useful for debugging and analysis; generally False for
        application-facing output.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1 inclusive.")
    validate_inference_data(claims)
    pipeline, metadata = load_model_artifact(model_path)

    # Feature engineering is kept identical to training and excludes target/IDs.
    features = engineer_features(claims, drop_identifiers=True, drop_target=True)

    # Explicitly align features with the saved training schema. This protects
    # inference from ordering differences and fails clearly for absent fields.
    expected_features = metadata.get("feature_columns")
    if expected_features:
        missing_features = [c for c in expected_features if c not in features.columns]
        extra_features = [c for c in features.columns if c not in expected_features]
        if missing_features:
            raise DataValidationError(
                f"Feature-engineering output is missing model fields: {missing_features}"
            )
        if extra_features:
            # Extras are ignored because model was trained on a fixed feature set.
            pass
        features = features.loc[:, expected_features]

    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = claims.copy().reset_index(drop=True)
    if "FraudFound_P" in output.columns:
        output = output.drop(columns=["FraudFound_P"])
    output["Fraud_Risk_Score"] = probabilities
    output["Fraud_Risk_Percent"] = probabilities * 100.0
    output["Predicted_Fraud"] = predictions
    output["Risk_Band"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.10, 0.30, 0.50, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
    ).astype(str)

    # If no identifier is supplied, create a row-level ID for this scoring run.
    if "PolicyNumber" not in output.columns:
        output.insert(0, "Scoring_Row_ID", np.arange(1, len(output) + 1))

    if include_features:
        engineered_output = features.reset_index(drop=True)
        output = pd.concat([output, engineered_output], axis=1)

    output.attrs["model_name"] = metadata.get("model_name", "Loaded fraud model")
    output.attrs["model_trained_at"] = metadata.get("trained_at_utc")
    output.attrs["threshold"] = threshold
    return output


def score_csv(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Load a CSV, score it, save scored output, and return the DataFrame."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    claims = pd.read_csv(input_path)
    scored = score_claims(claims, model_path=model_path, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print("\nSCORING COMPLETE")
    print("=" * 60)
    print(f"Input file:       {input_path}")
    print(f"Claims scored:    {len(scored):,}")
    print(f"Threshold:        {threshold:.2f}")
    print(f"Flagged claims:   {int(scored['Predicted_Fraud'].sum()):,}")
    print(f"Output file:      {output_path}")
    print("\nRisk-band counts:")
    print(scored["Risk_Band"].value_counts().sort_index().to_string())
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score incoming insurance claims for fraud risk.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input claims CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output scored CSV path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Model artifact path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fraud classification threshold")
    args = parser.parse_args()
    score_csv(args.input, args.output, model_path=args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()
