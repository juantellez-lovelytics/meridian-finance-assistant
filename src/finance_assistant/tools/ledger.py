"""R1 — deterministic ledger filtering.

`query_ledger` never does its own I/O: it takes an already-loaded,
already-validated GL DataFrame and returns a filtered slice plus the
metadata describing exactly what filter was applied — which date field
was used (accrual_date by default, never posting_date), the bounds, and
which dimensions were active. This is what lets the trace/evidence layer
show why a row was (or wasn't) included, per R1.
"""

from dataclasses import dataclass

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD

_DIMENSION_COLUMNS = {
    "entities": "entity",
    "cost_centres": "cost_centre",
    "account_codes": "account_code",
    "vendor_ids": "vendor_id",
}


@dataclass(frozen=True)
class LedgerQueryFilters:
    date_field: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    entities: list[str] | None
    cost_centres: list[str] | None
    account_codes: list[str] | None
    vendor_ids: list[str] | None


@dataclass(frozen=True)
class LedgerQueryResult:
    rows: pd.DataFrame
    filters: LedgerQueryFilters
    rows_in: int
    rows_matched: int


def query_ledger(
    rows: pd.DataFrame,
    date_start: str | pd.Timestamp,
    date_end: str | pd.Timestamp,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
    entities: list[str] | None = None,
    cost_centres: list[str] | None = None,
    account_codes: list[str] | None = None,
    vendor_ids: list[str] | None = None,
) -> LedgerQueryResult:
    if date_field not in rows.columns:
        raise ValueError(f"date_field '{date_field}' is not a column of the supplied rows")
    if not pd.api.types.is_datetime64_any_dtype(rows[date_field]):
        raise ValueError(f"date_field '{date_field}' is not a datetime column")

    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    if start > end:
        raise ValueError(f"date_start ({start}) is after date_end ({end})")

    dimension_values = {
        "entities": entities,
        "cost_centres": cost_centres,
        "account_codes": account_codes,
        "vendor_ids": vendor_ids,
    }

    mask = rows[date_field].between(start, end, inclusive="both")
    for dimension, values in dimension_values.items():
        if values is None:
            continue
        column = _DIMENSION_COLUMNS[dimension]
        if column not in rows.columns:
            raise ValueError(f"'{dimension}' filter requires column '{column}', not present in rows")
        mask &= rows[column].isin(values)

    matched = rows.loc[mask]

    filters = LedgerQueryFilters(
        date_field=date_field,
        date_start=start,
        date_end=end,
        entities=entities,
        cost_centres=cost_centres,
        account_codes=account_codes,
        vendor_ids=vendor_ids,
    )
    return LedgerQueryResult(
        rows=matched,
        filters=filters,
        rows_in=len(rows),
        rows_matched=len(matched),
    )
