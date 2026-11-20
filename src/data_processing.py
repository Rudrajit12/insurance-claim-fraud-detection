"""
Reusable data loading and validation utilities for Insurance Claim Fraud Detection.

Responsibilities:
- Load claims CSV files
- Validate expected columns and target values
- Separate predictors and target
- Drop identifier-like columns from predictors
- Provide a compact dataset audit
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

TARGET_COLUMN = "FraudFound_P"
IDENTIFIER_COLUMNS = ["PolicyNumber", "RepNumber"]

EXPECTED_COLUMNS = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea",
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Sex",
    "MaritalStatus", "Age", "Fault", "PolicyType", "VehicleCategory",
    "VehiclePrice", "FraudFound_P", "PolicyNumber", "RepNumber",
    "Deductible", "DriverRating", "Days_Policy_Accident",
    "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent",
    "AgentType", "NumberOfSuppliments", "AddressChange_Claim",
    "NumberOfCars", "Year", "BasePolicy",
]


class DataValidationError(ValueError):
    """Raised when input data fails required validation checks."""


def load_claims_csv(
    file_path: str | Path,
    *,
    validate_schema: bool = True,
    expected_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a claims CSV and optionally validate its schema."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    df = pd.read_csv(path)
    if validate_schema:
        validate_claims_schema(
            df,
            expected_columns=expected_columns or EXPECTED_COLUMNS,
            require_target=True,
        )
    return df


def validate_claims_schema(
    df: pd.DataFrame,
    *,
    expected_columns: Optional[Iterable[str]] = None,
    require_target: bool = True,
) -> None:
    """Check non-empty data, duplicate column names, required fields, and target."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise DataValidationError("Claims dataset is empty.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate column names detected: {duplicates}")

    required = list(expected_columns or EXPECTED_COLUMNS)
    if not require_target:
        required = [col for col in required if col != TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if require_target and TARGET_COLUMN in df.columns:
        values = set(df[TARGET_COLUMN].dropna().unique())
        if not values.issubset({0, 1}):
            raise DataValidationError(
                f"Target {TARGET_COLUMN} must contain only 0/1 values; found: {sorted(values)}"
            )


def separate_target(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    drop_identifiers: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X and y, optionally dropping identifier-like predictor columns."""
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' not found.")
    y = df[target_column].astype(int).copy()
    X = df.drop(columns=[target_column]).copy()
    if drop_identifiers:
        to_drop = [col for col in IDENTIFIER_COLUMNS if col in X.columns]
        X = X.drop(columns=to_drop)
    return X, y


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return a compact audit summary suitable for notebook or application use."""
    target_valid = TARGET_COLUMN in df.columns and df[TARGET_COLUMN].notna().all()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(len(df.select_dtypes(include="number").columns)),
        "categorical_columns": int(len(df.select_dtypes(exclude="number").columns)),
        "target_column_present": TARGET_COLUMN in df.columns,
        "fraud_count": int(df[TARGET_COLUMN].sum()) if target_valid else None,
        "fraud_rate": float(df[TARGET_COLUMN].mean()) if target_valid else None,
    }
