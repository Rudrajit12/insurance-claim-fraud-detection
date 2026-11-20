"""
Reusable feature engineering for Insurance Claim Fraud Detection.

The transformations mirror the logic validated in Notebook 03 and are intended
for consistent use in both model training and scoring.
"""

from typing import Optional

import numpy as np
import pandas as pd

VEHICLE_AGE_MAP = {
    "new": 0, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "more than 7": 8,
}
PAST_CLAIMS_MAP = {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 5}
SUPPLEMENT_MAP = {"none": 0, "1 to 2": 1.5, "3 to 5": 4, "more than 5": 6}
POLICY_ACCIDENT_MAP = {
    "none": 0, "1 to 7": 4, "8 to 15": 11.5,
    "15 to 30": 22.5, "more than 30": 31,
}
POLICY_CLAIM_MAP = {
    "none": 0, "8 to 15": 11.5, "15 to 30": 22.5, "more than 30": 31,
}
ADDRESS_CHANGE_MAP = {
    "under 6 months": 0.25, "1 year": 1, "2 to 3 years": 2.5, "4 to 8 years": 6,
}
VEHICLE_PRICE_MAP = {
    "less than 20000": 1, "20000 to 29000": 2, "30000 to 39000": 3,
    "40000 to 59000": 4, "60000 to 69000": 5, "more than 69000": 6,
}
NUMBER_CARS_MAP = {
    "1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3.5, "5 to 8": 6.5, "more than 8": 9,
}

IDENTIFIER_COLUMNS = ["PolicyNumber", "RepNumber"]
TARGET_COLUMN = "FraudFound_P"
REQUIRED_RAW_FEATURES = [
    "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed", "Age",
    "AgeOfVehicle", "PastNumberOfClaims", "NumberOfSuppliments",
    "Days_Policy_Accident", "Days_Policy_Claim", "AddressChange_Claim",
    "VehiclePrice", "NumberOfCars",
]


def _validate_feature_columns(df: pd.DataFrame, required: Optional[list] = None) -> None:
    required = required or REQUIRED_RAW_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for feature engineering: {missing}")


def engineer_features(
    claims: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
    drop_target: bool = True,
) -> pd.DataFrame:
    """Apply Notebook 03 transformations and return predictors.

    Target and identifier columns are removed by default to prevent leakage.
    Existing raw feature columns are retained alongside engineered features.
    """
    _validate_feature_columns(claims)
    X = claims.copy()

    if drop_target and TARGET_COLUMN in X.columns:
        X = X.drop(columns=[TARGET_COLUMN])
    if drop_identifiers:
        ids = [col for col in IDENTIFIER_COLUMNS if col in X.columns]
        X = X.drop(columns=ids)

    # Historical placeholder "0" means unknown for these claimed date fields.
    for col in ["MonthClaimed", "DayOfWeekClaimed"]:
        if col in X.columns:
            X[col] = X[col].replace("0", np.nan)

    X["AgeGroup"] = pd.cut(
        X["Age"],
        bins=[-1, 20, 25, 30, 35, 40, 50, 65, np.inf],
        labels=["<=20", "21-25", "26-30", "31-35", "36-40", "41-50", "51-65", "65+"],
        right=True,
    ).astype("object")

    X["AccidentClaimMonthMismatch"] = np.where(
        X["MonthClaimed"].notna(), (X["Month"] != X["MonthClaimed"]).astype(int), np.nan
    )
    X["AccidentClaimDayMismatch"] = np.where(
        X["DayOfWeekClaimed"].notna(), (X["DayOfWeek"] != X["DayOfWeekClaimed"]).astype(int), np.nan
    )

    X["VehicleAgeYearsOrdinal"] = X["AgeOfVehicle"].map(VEHICLE_AGE_MAP)
    X["PastClaimsOrdinal"] = X["PastNumberOfClaims"].map(PAST_CLAIMS_MAP)
    X["SupplementCountOrdinal"] = X["NumberOfSuppliments"].map(SUPPLEMENT_MAP)
    X["DaysPolicyAccidentOrdinal"] = X["Days_Policy_Accident"].map(POLICY_ACCIDENT_MAP)
    X["DaysPolicyClaimOrdinal"] = X["Days_Policy_Claim"].map(POLICY_CLAIM_MAP)
    X["AddressChangeYearsOrdinal"] = X["AddressChange_Claim"].map(ADDRESS_CHANGE_MAP)
    X["AddressChangedFlag"] = (X["AddressChange_Claim"] != "no change").astype(int)
    X["VehiclePriceOrdinal"] = X["VehiclePrice"].map(VEHICLE_PRICE_MAP)
    X["NumberOfCarsOrdinal"] = X["NumberOfCars"].map(NUMBER_CARS_MAP)

    return X


def engineered_feature_names() -> list[str]:
    """List the feature columns added by engineer_features()."""
    return [
        "AgeGroup", "AccidentClaimMonthMismatch", "AccidentClaimDayMismatch",
        "VehicleAgeYearsOrdinal", "PastClaimsOrdinal", "SupplementCountOrdinal",
        "DaysPolicyAccidentOrdinal", "DaysPolicyClaimOrdinal",
        "AddressChangeYearsOrdinal", "AddressChangedFlag", "VehiclePriceOrdinal",
        "NumberOfCarsOrdinal",
    ]
