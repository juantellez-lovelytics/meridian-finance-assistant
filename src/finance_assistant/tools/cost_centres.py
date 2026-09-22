"""R4 — cost-centre reporting normalization, temporal and non-destructive.

The board memo (see config.COST_CENTRE_REPORTING_TRANSITIONS) documents a
mid-year cost-centre code change — OPS-NA reported as OPS-AMER effective
2024-07-01 — with the full-year plan restated onto the new code but prior
GL comparatives left on the old one. The transition table itself is read
from config (sourced from the memo); this module never infers or
hardcodes the mapping.

`normalize_reporting_cost_centre` never overwrites `cost_centre`: it adds
`source_cost_centre` (an untouched copy) and `reporting_cost_centre` (the
code to use once a comparison spans the effective date). Only rows dated
before a transition's effective_date and still carrying the source code
get remapped — rows already on the reporting code are left alone. Every
row actually remapped is recorded in `applied`, citing the document and
section that justifies it.
"""

from dataclasses import dataclass
from typing import Hashable

import pandas as pd

from finance_assistant.config import (
    COST_CENTRE_REPORTING_TRANSITIONS,
    DEFAULT_FINANCIAL_DATE_FIELD,
    CostCentreTransition,
)


@dataclass(frozen=True)
class AppliedNormalization:
    row_index: Hashable
    source_cost_centre: str
    reporting_cost_centre: str
    effective_date: pd.Timestamp
    source_document: str
    source_section: str
    row_id: str | None = None


@dataclass(frozen=True)
class CostCentreNormalizationResult:
    rows: pd.DataFrame
    date_field: str
    total_rows: int
    applied: list[AppliedNormalization]

    @property
    def normalized_rows(self) -> int:
        return len(self.applied)


def normalize_reporting_cost_centre(
    rows: pd.DataFrame,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
    transitions: list[CostCentreTransition] | None = None,
) -> CostCentreNormalizationResult:
    if transitions is None:
        transitions = COST_CENTRE_REPORTING_TRANSITIONS

    if date_field not in rows.columns:
        raise ValueError(f"date_field '{date_field}' is not a column of the supplied rows")
    if not pd.api.types.is_datetime64_any_dtype(rows[date_field]):
        raise ValueError(f"date_field '{date_field}' is not a datetime column")
    if "cost_centre" not in rows.columns:
        raise ValueError("rows must have a 'cost_centre' column")

    working = rows.copy()
    working["source_cost_centre"] = working["cost_centre"]
    working["reporting_cost_centre"] = working["cost_centre"]

    has_txn_id = "txn_id" in working.columns
    applied: list[AppliedNormalization] = []
    for transition in transitions:
        effective_date = pd.Timestamp(transition["effective_date"])
        mask = (working["cost_centre"] == transition["source_cost_centre"]) & (
            working[date_field] < effective_date
        )
        matched_index = working.index[mask]
        if len(matched_index) == 0:
            continue

        working.loc[matched_index, "reporting_cost_centre"] = transition["reporting_cost_centre"]
        for row_index in matched_index:
            applied.append(
                AppliedNormalization(
                    row_index=row_index,
                    source_cost_centre=transition["source_cost_centre"],
                    reporting_cost_centre=transition["reporting_cost_centre"],
                    effective_date=effective_date,
                    source_document=transition["source_document"],
                    source_section=transition["source_section"],
                    row_id=working.at[row_index, "txn_id"] if has_txn_id else None,
                )
            )

    return CostCentreNormalizationResult(
        rows=working,
        date_field=date_field,
        total_rows=len(rows),
        applied=applied,
    )
