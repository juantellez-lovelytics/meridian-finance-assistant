"""Tests for finance_assistant.tools.cost_centres.normalize_reporting_cost_centre (R4).

Concrete values live only in synthetic tmp_path fixtures with a synthetic
transition table (CC-OLD -> CC-NEW), never the real dataset's cost-centre
codes baked into assertions. Only the structural checks at the bottom
touch config.COST_CENTRE_REPORTING_TRANSITIONS (the real, document-derived
table) and the real data/, and only to confirm shape/invariants, never a
hardcoded count or amount.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_gl_transactions
from finance_assistant.tools.cost_centres import normalize_reporting_cost_centre

_SYNTHETIC_TRANSITIONS = [
    {
        "source_cost_centre": "CC-OLD",
        "reporting_cost_centre": "CC-NEW",
        "effective_date": "2024-07-01",
        "source_document": "synthetic_memo.md",
        "source_section": "1. Synthetic reorg",
    }
]


def _gl_row(**overrides):
    row = {
        "txn_id": "T0001",
        "posting_date": "2024-06-05",
        "accrual_date": "2024-06-01",
        "entity": "MI-US",
        "cost_centre": "CC-OLD",
        "account_code": "9000",
        "amount": "100.00",
        "currency": "USD",
        "vendor_id": "V1001",
        "doc_ref": "INV-0001",
        "approval_ref": "",
        "memo": "test row",
    }
    row.update(overrides)
    return row


def _load_gl(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


# ---------------------------------------------------------------------------
# Core remapping behavior.
# ---------------------------------------------------------------------------


def test_row_before_effective_date_is_remapped(write_csv):
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_BEFORE", cost_centre="CC-OLD", accrual_date="2024-03-01")])

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    row = result.rows.set_index("txn_id").loc["T_BEFORE"]
    assert row["source_cost_centre"] == "CC-OLD"
    assert row["reporting_cost_centre"] == "CC-NEW"
    assert result.normalized_rows == 1
    assert result.applied[0].source_document == "synthetic_memo.md"
    assert result.applied[0].source_section == "1. Synthetic reorg"
    assert result.applied[0].row_id == "T_BEFORE"


def test_row_already_on_reporting_code_is_left_alone(write_csv):
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_AFTER", cost_centre="CC-NEW", accrual_date="2024-09-01")])

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    row = result.rows.set_index("txn_id").loc["T_AFTER"]
    assert row["source_cost_centre"] == "CC-NEW"
    assert row["reporting_cost_centre"] == "CC-NEW"
    assert result.normalized_rows == 0


def test_unrelated_cost_centre_is_untouched(write_csv):
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_OTHER", cost_centre="ENG-US", accrual_date="2024-03-01")])

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    row = result.rows.set_index("txn_id").loc["T_OTHER"]
    assert row["source_cost_centre"] == "ENG-US"
    assert row["reporting_cost_centre"] == "ENG-US"
    assert result.normalized_rows == 0


def test_source_value_is_never_overwritten_on_the_raw_column(write_csv):
    gl = _load_gl(write_csv, [_gl_row(cost_centre="CC-OLD", accrual_date="2024-01-01")])

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    assert result.rows["cost_centre"].iloc[0] == "CC-OLD"
    assert result.rows["source_cost_centre"].iloc[0] == "CC-OLD"


def test_effective_date_boundary_is_exclusive(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_ON_BOUNDARY", cost_centre="CC-OLD", accrual_date="2024-07-01"),
            _gl_row(txn_id="T_JUST_BEFORE", cost_centre="CC-OLD", accrual_date="2024-06-30"),
        ],
    )

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    by_txn = result.rows.set_index("txn_id")
    assert by_txn.loc["T_JUST_BEFORE", "reporting_cost_centre"] == "CC-NEW"
    # A row still on the source code on/after the effective date is a data
    # anomaly the transition rule doesn't cover -- left as-is rather than
    # silently forced onto the new code.
    assert by_txn.loc["T_ON_BOUNDARY", "reporting_cost_centre"] == "CC-OLD"


def test_multiple_matching_rows_all_recorded_in_applied(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id=f"T{i:04d}", cost_centre="CC-OLD", accrual_date="2024-02-01")
            for i in range(3)
        ],
    )

    result = normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)

    assert result.normalized_rows == 3
    assert {a.row_id for a in result.applied} == {"T0000", "T0001", "T0002"}


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_missing_cost_centre_column_raises(write_csv):
    gl = _load_gl(write_csv, [_gl_row()]).drop(columns=["cost_centre"])

    with pytest.raises(ValueError):
        normalize_reporting_cost_centre(gl, transitions=_SYNTHETIC_TRANSITIONS)


def test_unknown_date_field_raises(write_csv):
    gl = _load_gl(write_csv, [_gl_row()])

    with pytest.raises(ValueError):
        normalize_reporting_cost_centre(gl, date_field="not_a_column", transitions=_SYNTHETIC_TRANSITIONS)


# ---------------------------------------------------------------------------
# Structural checks against the real, document-derived transition table.
# ---------------------------------------------------------------------------


def test_real_transition_table_shape():
    assert len(config.COST_CENTRE_REPORTING_TRANSITIONS) > 0
    for transition in config.COST_CENTRE_REPORTING_TRANSITIONS:
        assert transition["source_cost_centre"] != transition["reporting_cost_centre"]
        assert transition["source_document"]
        assert transition["source_section"]
        pd.Timestamp(transition["effective_date"])  # must parse as a date


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()

    result = normalize_reporting_cost_centre(gl)

    assert result.total_rows == len(gl)
    assert all(a.source_cost_centre != a.reporting_cost_centre for a in result.applied)

    known_codes = set(gl["cost_centre"].unique()) | {
        t["reporting_cost_centre"] for t in config.COST_CENTRE_REPORTING_TRANSITIONS
    }
    assert set(result.rows["reporting_cost_centre"].unique()) <= known_codes
    assert set(result.rows["source_cost_centre"].unique()) == set(gl["cost_centre"].unique())
