"""Business decision engine for insurance claim fraud investigation prioritization.

The Risk Engine consumes scored claims (normally produced by src.predict.score_claims),
orders claims by fraud risk, selects a capacity-constrained investigation queue, assigns
operational priority labels, and produces a compact summary.

Important: priority bands are operational display labels, not validated fraud rules.
The default investigation capacity is 200 claims per 10,000 incoming claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_CAPACITY_PER_10K = 200
SCORE_COLUMN = "Fraud_Risk_Score"


def _validate_scored_claims(claims: pd.DataFrame, score_column: str = SCORE_COLUMN) -> None:
    if not isinstance(claims, pd.DataFrame):
        raise TypeError("claims must be a pandas DataFrame")
    if claims.empty:
        raise ValueError("claims dataframe is empty")
    if score_column not in claims.columns:
        raise ValueError(f"Required score column '{score_column}' is missing")
    scores = pd.to_numeric(claims[score_column], errors="coerce")
    if scores.isna().any():
        raise ValueError(f"'{score_column}' contains missing or non-numeric values")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"'{score_column}' values must be between 0 and 1")


def calculate_capacity(
    total_claims: int,
    capacity: Optional[int] = None,
    capacity_per_10k: int = DEFAULT_CAPACITY_PER_10K,
) -> int:
    """Calculate investigation capacity, scaling 200 per 10,000 if not overridden."""
    if total_claims < 0:
        raise ValueError("total_claims must be non-negative")
    if capacity is not None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        return min(int(capacity), int(total_claims))
    if capacity_per_10k < 0:
        raise ValueError("capacity_per_10k must be non-negative")
    # Round to nearest whole claim, with at least one slot for a non-empty set if
    # capacity rate is positive and the proportional result is below one.
    proportional = total_claims * capacity_per_10k / 10_000
    return min(int(total_claims), int(round(proportional)))


def assign_priority(
    scores: pd.Series,
    *,
    high_threshold: float = 0.50,
    medium_threshold: float = 0.30,
    low_threshold: float = 0.10,
) -> pd.Series:
    """Assign display priorities based on probability-score thresholds."""
    if not (0 <= low_threshold <= medium_threshold <= high_threshold <= 1):
        raise ValueError("Priority thresholds must satisfy 0 <= low <= medium <= high <= 1")
    values = pd.to_numeric(scores, errors="coerce")
    labels = pd.Series("Low", index=scores.index, dtype="object")
    labels.loc[values >= low_threshold] = "Moderate"
    labels.loc[values >= medium_threshold] = "High"
    labels.loc[values >= high_threshold] = "Critical"
    return labels


def build_investigation_queue(
    scored_claims: pd.DataFrame,
    *,
    capacity: Optional[int] = None,
    capacity_per_10k: int = DEFAULT_CAPACITY_PER_10K,
    score_column: str = SCORE_COLUMN,
    claim_id_column: Optional[str] = None,
    high_threshold: float = 0.50,
    medium_threshold: float = 0.30,
    low_threshold: float = 0.10,
) -> pd.DataFrame:
    """Rank claims and mark the capacity-constrained investigation queue.

    Returns all claims sorted by descending score. Queue_Position is populated only
    for selected claims; Investigation_Selected is a boolean flag. A stable sort
    preserves original row order for ties.
    """
    _validate_scored_claims(scored_claims, score_column)
    n = len(scored_claims)
    k = calculate_capacity(n, capacity=capacity, capacity_per_10k=capacity_per_10k)

    ranked = scored_claims.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="raise")
    ranked["Risk_Rank"] = ranked[score_column].rank(method="first", ascending=False).astype(int)
    ranked = ranked.sort_values(score_column, ascending=False, kind="mergesort").reset_index(drop=True)
    ranked["Risk_Rank"] = range(1, n + 1)
    ranked["Priority"] = assign_priority(
        ranked[score_column],
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
    )
    ranked["Investigation_Selected"] = ranked["Risk_Rank"] <= k
    ranked["Queue_Position"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked.loc[ranked["Investigation_Selected"], "Queue_Position"] = ranked.loc[
        ranked["Investigation_Selected"], "Risk_Rank"
    ].astype("Int64")
    ranked["Investigation_Status"] = "Not Selected"
    ranked.loc[ranked["Investigation_Selected"], "Investigation_Status"] = "Pending"

    # Add claim identifier if caller provides one; no ID is invented if absent.
    if claim_id_column and claim_id_column not in ranked.columns:
        raise ValueError(f"claim_id_column '{claim_id_column}' was not found")
    return ranked


def summarize_queue(
    queue: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
) -> dict:
    """Return capacity, risk-score, priority, and selection summary statistics."""
    _validate_scored_claims(queue, score_column)
    selected = queue[queue["Investigation_Selected"]]
    counts = queue["Priority"].value_counts().to_dict()
    selected_priority_counts = selected["Priority"].value_counts().to_dict()
    return {
        "total_claims": int(len(queue)),
        "selected_for_investigation": int(len(selected)),
        "capacity_utilization_pct": float(len(selected) / len(queue) * 100) if len(queue) else 0.0,
        "mean_risk_score_all": float(queue[score_column].mean()),
        "mean_risk_score_selected": float(selected[score_column].mean()) if len(selected) else None,
        "max_risk_score_selected": float(selected[score_column].max()) if len(selected) else None,
        "min_risk_score_selected": float(selected[score_column].min()) if len(selected) else None,
        "priority_counts_all": {str(k): int(v) for k, v in counts.items()},
        "priority_counts_selected": {str(k): int(v) for k, v in selected_priority_counts.items()},
    }


def export_queue(
    queue: pd.DataFrame,
    output_path: str | Path,
    *,
    selected_only: bool = True,
) -> Path:
    """Export the investigation queue to CSV; selected-only is the default."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = queue[queue["Investigation_Selected"]].copy() if selected_only else queue.copy()
    output.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Create a capacity-constrained fraud investigation queue.")
    parser.add_argument("input", help="CSV containing Fraud_Risk_Score")
    parser.add_argument("--output", default="reports/investigation_queue.csv")
    parser.add_argument("--capacity", type=int, default=None, help="Absolute capacity override")
    parser.add_argument("--capacity-per-10k", type=int, default=DEFAULT_CAPACITY_PER_10K)
    parser.add_argument("--all-ranked", action="store_true", help="Export all ranked claims, not selected-only")
    args = parser.parse_args()

    claims = pd.read_csv(args.input)
    queue = build_investigation_queue(
        claims,
        capacity=args.capacity,
        capacity_per_10k=args.capacity_per_10k,
    )
    output_path = export_queue(queue, args.output, selected_only=not args.all_ranked)
    print(json.dumps(summarize_queue(queue), indent=2))
    print(f"Exported: {output_path}")
