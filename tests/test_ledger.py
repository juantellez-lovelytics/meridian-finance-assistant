"""Tests for finance_assistant.tools.ledger.query_ledger (R1).

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loader, never against the real data/*.csv
files (see tests/test_loaders.py for the convention and rationale).
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_gl_transactions
from finance_assistant.tools.ledger import query_ledger


def _gl_row(**overrides):
    row = {
        "txn_id": "T0001",
        "posting_date": "2024-01-05",
        "accrual_date": "2024-01-01",
        "entity": "MI-US",
        "cost_centre": "OPS-NA",
        "account_code": "6320",
        "amount": "100.00",
        "currency": "USD",
        "vendor_id": "V1001",
        "doc_ref": "INV-0001",
        "approval_ref": "",
        "memo": "test row",
    }
    row.update(overrides)
    return row


def _load(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


# ---------------------------------------------------------------------------
# R1: accrual_date is the default financial date basis, never posting_date.
# ---------------------------------------------------------------------------


def test_default_date_field_uses_accrual_date_across_year_boundary(write_csv):
    """Row accrued in 2024, posted in 2025 -> belongs to FY2024 under the default."""
    rows = [
        _gl_row(txn_id="T_STRADDLE", accrual_date="2024-12-30", posting_date="2025-01-05"),
        _gl_row(txn_id="T_OTHER_YEAR", accrual_date="2025-02-01", posting_date="2025-02-03"),
    ]
    gl = _load(write_csv, rows)

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31")

    assert set(result.rows["txn_id"]) == {"T_STRADDLE"}
    assert result.filters.date_field == config.DEFAULT_FINANCIAL_DATE_FIELD


def test_posting_date_field_excludes_the_same_straddling_row(write_csv):
    rows = [_gl_row(txn_id="T_STRADDLE", accrual_date="2024-12-30", posting_date="2025-01-05")]
    gl = _load(write_csv, rows)

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31", date_field="posting_date")

    assert set(result.rows["txn_id"]) == set()
    assert result.filters.date_field == "posting_date"


# ---------------------------------------------------------------------------
# Date bounds.
# ---------------------------------------------------------------------------


def test_date_bounds_are_inclusive(write_csv):
    rows = [
        _gl_row(txn_id="T_START", accrual_date="2024-01-01"),
        _gl_row(txn_id="T_END", accrual_date="2024-12-31"),
        _gl_row(txn_id="T_BEFORE", accrual_date="2023-12-31"),
        _gl_row(txn_id="T_AFTER", accrual_date="2025-01-01"),
    ]
    gl = _load(write_csv, rows)

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31")

    assert set(result.rows["txn_id"]) == {"T_START", "T_END"}


def test_date_start_after_date_end_raises(write_csv):
    gl = _load(write_csv, [_gl_row()])

    with pytest.raises(ValueError):
        query_ledger(gl, date_start="2024-12-31", date_end="2024-01-01")


def test_unknown_date_field_raises(write_csv):
    gl = _load(write_csv, [_gl_row()])

    with pytest.raises(ValueError):
        query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31", date_field="not_a_column")


# ---------------------------------------------------------------------------
# Dimension filters.
# ---------------------------------------------------------------------------


def test_dimension_filters_independently_and_combined(write_csv):
    rows = [
        _gl_row(txn_id="T_A", entity="MI-US", cost_centre="OPS-NA", account_code="6320", vendor_id="V1001"),
        _gl_row(txn_id="T_B", entity="MI-NL", cost_centre="OPS-EU", account_code="6130", vendor_id="V1002"),
        _gl_row(txn_id="T_C", entity="MI-US", cost_centre="SGA-NA", account_code="6320", vendor_id="V1003"),
    ]
    gl = _load(write_csv, rows)
    date_kwargs = {"date_start": "2024-01-01", "date_end": "2024-12-31"}

    assert set(query_ledger(gl, entities=["MI-US"], **date_kwargs).rows["txn_id"]) == {"T_A", "T_C"}
    assert set(query_ledger(gl, cost_centres=["OPS-EU"], **date_kwargs).rows["txn_id"]) == {"T_B"}
    assert set(query_ledger(gl, account_codes=["6320"], **date_kwargs).rows["txn_id"]) == {"T_A", "T_C"}
    assert set(query_ledger(gl, vendor_ids=["V1002"], **date_kwargs).rows["txn_id"]) == {"T_B"}

    combined = query_ledger(gl, entities=["MI-US"], account_codes=["6320"], **date_kwargs)
    assert set(combined.rows["txn_id"]) == {"T_A", "T_C"}

    narrowed = query_ledger(gl, entities=["MI-US"], cost_centres=["SGA-NA"], **date_kwargs)
    assert set(narrowed.rows["txn_id"]) == {"T_C"}


def test_none_dimension_leaves_it_unfiltered(write_csv):
    rows = [_gl_row(txn_id="T_A", entity="MI-US"), _gl_row(txn_id="T_B", entity="MI-NL")]
    gl = _load(write_csv, rows)

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31", entities=None)

    assert set(result.rows["txn_id"]) == {"T_A", "T_B"}


def test_missing_dimension_column_raises(write_csv):
    gl = _load(write_csv, [_gl_row()]).drop(columns=["vendor_id"])

    with pytest.raises(ValueError):
        query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31", vendor_ids=["V1001"])


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_result_metadata_reflects_applied_filter(write_csv):
    rows = [_gl_row(txn_id=f"T{i:04d}") for i in range(3)]
    gl = _load(write_csv, rows)

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31", entities=["MI-US"])

    assert result.filters.date_field == config.DEFAULT_FINANCIAL_DATE_FIELD
    assert result.filters.date_start == pd.Timestamp("2024-01-01")
    assert result.filters.date_end == pd.Timestamp("2024-12-31")
    assert result.filters.entities == ["MI-US"]
    assert result.filters.cost_centres is None
    assert result.rows_in == 3
    assert result.rows_matched == 3


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()

    result = query_ledger(gl, date_start="2024-01-01", date_end="2024-12-31")

    assert result.rows_matched <= result.rows_in
    assert set(result.rows["txn_id"]) <= set(gl["txn_id"])
