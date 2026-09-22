"""R3 — strict temporal chart-of-accounts join.

A row's account_code can map to different parents in different periods
(the COA carries valid_from/valid_to vigencia). The join here is always
filtered by vigencia before anything else happens: no
`drop_duplicates("account_code")`, no `set_index("account_code")`, no
picking "the first match" silently.

Ambiguity (a row landing inside more than one COA validity window — a
data-quality defect in the COA itself) is always a hard error, regardless
of `strict`. Zero-mapping is degradable via `strict=False`: the Evidence
Gate, not this tool, decides whether an incomplete mapping is PARTIAL or
REFUSED, so the tool must be able to hand back a coverage figure instead
of blowing up the whole call over one orphaned row.
"""

from dataclasses import dataclass
from typing import Hashable

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD

_REQUIRED_COA_COLUMNS = [
    "account_code",
    "account_name",
    "parent_code",
    "parent_name",
    "statement_line",
    "valid_from",
    "valid_to",
]
_COA_STRING_COLUMNS = ["account_name", "parent_code", "parent_name", "statement_line"]
_COA_DATE_COLUMNS = ["valid_from", "valid_to"]


@dataclass(frozen=True)
class AccountMappingIssue:
    row_index: Hashable
    account_code: str
    financial_date: pd.Timestamp
    match_count: int  # 0 = unmapped, >1 = ambiguous
    row_id: str | None = None


class AccountMappingError(Exception):
    def __init__(self, date_field: str, issues: list[AccountMappingIssue]) -> None:
        self.date_field = date_field
        self.issues = issues
        self.unmapped = [i for i in issues if i.match_count == 0]
        self.ambiguous = [i for i in issues if i.match_count > 1]
        super().__init__(
            f"account hierarchy resolution failed on date_field '{date_field}': "
            f"{len(self.unmapped)} row(s) with zero COA match, "
            f"{len(self.ambiguous)} row(s) with ambiguous (>1) COA match"
        )


@dataclass(frozen=True)
class AccountHierarchyResult:
    rows: pd.DataFrame
    matched_rows: int
    total_rows: int
    date_field: str
    unmapped: list[AccountMappingIssue]

    @property
    def is_complete(self) -> bool:
        return self.matched_rows == self.total_rows


def resolve_account_hierarchy(
    rows: pd.DataFrame,
    coa: pd.DataFrame,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
    strict: bool = True,
) -> AccountHierarchyResult:
    if date_field not in rows.columns:
        raise ValueError(f"date_field '{date_field}' is not a column of the supplied rows")
    if not pd.api.types.is_datetime64_any_dtype(rows[date_field]):
        raise ValueError(f"date_field '{date_field}' is not a datetime column")
    if "account_code" not in rows.columns:
        raise ValueError("rows must have an 'account_code' column")
    missing_coa_columns = [c for c in _REQUIRED_COA_COLUMNS if c not in coa.columns]
    if missing_coa_columns:
        raise ValueError(f"coa is missing required column(s): {', '.join(missing_coa_columns)}")

    working = rows.copy()
    working["_row_index"] = working.index

    merged = working.merge(coa, on="account_code", how="left")
    in_window = (
        merged["valid_from"].notna()
        & merged[date_field].notna()
        & (merged[date_field] >= merged["valid_from"])
        & (merged[date_field] <= merged["valid_to"])
    )
    matched = merged.loc[in_window]

    match_counts = matched.groupby("_row_index").size().rename("match_count").reset_index()
    counts_by_row = working[["_row_index"]].merge(match_counts, on="_row_index", how="left")
    counts_by_row["match_count"] = counts_by_row["match_count"].fillna(0).astype(int)

    working_with_counts = working.assign(match_count=counts_by_row["match_count"].values)
    has_txn_id = "txn_id" in working.columns
    problem_rows = working_with_counts.loc[working_with_counts["match_count"] != 1]
    issues = [
        AccountMappingIssue(
            row_index=record["_row_index"],
            account_code=record["account_code"],
            financial_date=record[date_field],
            match_count=int(record["match_count"]),
            row_id=record["txn_id"] if has_txn_id else None,
        )
        for record in problem_rows.to_dict("records")
    ]
    ambiguous = [i for i in issues if i.match_count > 1]
    unmapped = [i for i in issues if i.match_count == 0]

    if ambiguous:
        raise AccountMappingError(date_field, issues=ambiguous + unmapped)
    if unmapped and strict:
        raise AccountMappingError(date_field, issues=unmapped)

    matched_ok = matched.copy()
    matched_ok["is_account_mapped"] = True

    unmapped_ids = [i.row_index for i in unmapped]
    unmapped_frame = working[working["_row_index"].isin(unmapped_ids)].copy()
    for col in _COA_STRING_COLUMNS:
        unmapped_frame[col] = pd.array([pd.NA] * len(unmapped_frame), dtype="string")
    for col in _COA_DATE_COLUMNS:
        # Match coa's datetime64[s] resolution explicitly: a default NaT
        # assignment creates datetime64[ns], and concatenating that against
        # coa's [s]-resolution 9999-12-31 sentinel overflows ns range.
        unmapped_frame[col] = pd.Series(pd.NaT, index=unmapped_frame.index, dtype="datetime64[s]")
    unmapped_frame["is_account_mapped"] = False

    combined = pd.concat([matched_ok, unmapped_frame], ignore_index=True)
    combined = combined.sort_values("_row_index").set_index("_row_index")
    combined.index.name = rows.index.name

    return AccountHierarchyResult(
        rows=combined,
        matched_rows=len(matched_ok),
        total_rows=len(rows),
        date_field=date_field,
        unmapped=unmapped,
    )
