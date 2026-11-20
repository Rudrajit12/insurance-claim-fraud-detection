"""Unit and integration checks for the insurance fraud pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing import DataValidationError, load_claims_csv, separate_target
from src.feature_engineering import engineer_features
from src.predict import score_claims, validate_inference_data
from src.risk_engine import build_investigation_queue, calculate_capacity, summarize_queue

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "fraud_oracle.csv"
MODEL_PATH = ROOT / "models" / "fraud_model_pipeline.joblib"


@pytest.fixture(scope="module")
def claims():
    return load_claims_csv(DATA_PATH)


def test_data_loading_and_schema(claims):
    assert len(claims) == 15420
    assert claims.shape[1] == 33
    assert claims["FraudFound_P"].isin([0, 1]).all()


def test_target_separation_drops_identifiers(claims):
    X, y = separate_target(claims)
    assert len(X) == len(y) == len(claims)
    assert "FraudFound_P" not in X.columns
    assert "PolicyNumber" not in X.columns
    assert "RepNumber" not in X.columns


def test_feature_engineering_schema_and_leakage(claims):
    engineered = engineer_features(claims)
    assert engineered.shape == (len(claims), 42)
    assert "FraudFound_P" not in engineered.columns
    assert "PolicyNumber" not in engineered.columns
    assert "RepNumber" not in engineered.columns
    assert "AgeGroup" in engineered.columns
    assert "VehiclePriceOrdinal" in engineered.columns


def test_inference_validation_rejects_missing_fields(claims):
    broken = claims.drop(columns=["Make"])
    with pytest.raises(DataValidationError, match="Missing required inference columns"):
        validate_inference_data(broken)


def test_scoring_outputs_valid_probability_and_no_target(claims):
    sample = claims.head(25)
    scored = score_claims(sample, model_path=MODEL_PATH)
    assert len(scored) == len(sample)
    assert "FraudFound_P" not in scored.columns
    assert scored["Fraud_Risk_Score"].between(0, 1).all()
    assert scored["Predicted_Fraud"].isin([0, 1]).all()
    assert scored["Risk_Band"].isin(["Low", "Moderate", "High", "Very High"]).all()


def test_capacity_calculation_modes():
    assert calculate_capacity(15420, capacity=200) == 200
    assert calculate_capacity(15420, capacity_per_10k=200) == 308
    assert calculate_capacity(100, capacity=200) == 100
    assert calculate_capacity(0, capacity=200) == 0


def test_queue_respects_capacity_and_ranking(claims):
    scored = score_claims(claims.head(300), model_path=MODEL_PATH)
    queue = build_investigation_queue(scored, capacity=20, claim_id_column="PolicyNumber")
    assert len(queue) == 300
    assert queue["Investigation_Selected"].sum() == 20
    assert queue["Fraud_Risk_Score"].is_monotonic_decreasing
    assert queue.iloc[:20]["Investigation_Selected"].all()
    positions = queue.loc[queue["Investigation_Selected"], "Queue_Position"].astype(int)
    assert positions.tolist() == list(range(1, 21))
    summary = summarize_queue(queue)
    assert summary["selected_for_investigation"] == 20


def test_risk_engine_rejects_invalid_scores():
    bad = pd.DataFrame({"Fraud_Risk_Score": [0.2, 1.1]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_investigation_queue(bad, capacity=1)
