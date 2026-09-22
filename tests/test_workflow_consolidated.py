"""Tests for finance_assistant.workflows.consolidated.consolidated_spend (Q3).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected totals are hand-computed, never derived by
calling the function under test.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_fx_rates, load_gl_transactions
from finance_assistant.evidence.models import AnswerStatus, MissingEvidenceReasonCode
from finance_assistant.workflows.consolidated import consolidated_spend


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-07-05",
        "accrual_date": "2024-07-05",
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "amount": "1000.00",
        "currency": "USD",
        "vendor_id": "",
        "doc_ref": "D1",
        "approval_ref": "",
        "memo": "",
    }
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-07", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, fx_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows)
    return load_gl_transactions(gl_path), load_fx_rates(fx_path)


def test_fx_gap_forces_refused_with_calculations_populated(write_csv):
    gl, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", entity="E1", currency="USD", amount="1000.00"),
            _gl_row(txn_id="T2", entity="E2", currency="EUR", amount="500.00"),
        ],
        [_fx_row(currency="USD")],  # no EUR rate at all
    )

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.result is None
    assert bundle.refusal_reason is not None

    by_description = {c.description: c.output for c in bundle.calculations}
    assert by_description["computable USD total (FX-convertible rows only)"] == pytest.approx(1000.0)
    assert by_description["computable components by entity (USD)"]["E1"] == pytest.approx(1000.0)
    assert by_description["non-convertible local amount by currency"]["EUR"] == pytest.approx(500.0)

    assert len(bundle.missing_evidence) == 1
    assert bundle.missing_evidence[0].reason_code == MissingEvidenceReasonCode.MISSING_FX_RATE


def test_full_fx_coverage_answers_with_exact_total(write_csv):
    gl, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", entity="E1", amount="1000.00"),
            _gl_row(txn_id="T2", entity="E2", amount="500.00"),
        ],
        [_fx_row()],
    )

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result["exact_total_usd"] == pytest.approx(1500.0)


def test_year_none_with_correlated_refusal_does_not_need_clarification(write_csv):
    # Both years have the same kind of FX gap -> both readings are REFUSED,
    # same status -> no status divergence -> gate should not force
    # NEEDS_CLARIFICATION purely from a shared refusal.
    gl, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-07-05", accrual_date="2024-07-05", currency="EUR", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-07-05", accrual_date="2023-07-05", currency="EUR", amount="1000.00"),
        ],
        [],  # no EUR rate for either year
    )

    bundle = consolidated_spend(gl, fx, "Q3", year=None)

    assert bundle.status == AnswerStatus.REFUSED


def test_year_none_with_materially_different_years_needs_clarification(write_csv):
    gl, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2023-07-05", accrual_date="2023-07-05", amount="100.00"),
            _gl_row(txn_id="T2", posting_date="2024-07-05", accrual_date="2024-07-05", amount="10000.00"),
        ],
        [_fx_row(period_month="2023-07"), _fx_row(period_month="2024-07")],
    )

    bundle = consolidated_spend(gl, fx, "Q3", year=None)

    assert bundle.status == AnswerStatus.NEEDS_CLARIFICATION
    assert bundle.result is None
    # No year was actually defaulted -- the gate rejected the default, so
    # the bundle must not claim it applied one.
    assert not any("defaulted" in a for a in bundle.assumptions)


def test_explicit_year_with_no_rows_refuses_via_no_data(write_csv):
    gl, fx = _load(write_csv, [_gl_row(posting_date="2024-07-05", accrual_date="2024-07-05")], [_fx_row()])

    bundle = consolidated_spend(gl, fx, "Q3", year=2099)

    assert bundle.status == AnswerStatus.REFUSED
    assert "no ledger rows" in bundle.refusal_reason


def test_tool_calls_are_recorded(write_csv):
    gl, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", entity="E1", amount="1000.00"),
            _gl_row(txn_id="T2", entity="E2", amount="500.00"),
        ],
        [_fx_row()],
    )

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {"query_ledger", "convert_to_usd", "aggregate_usd", "aggregate_usd_by"}
    assert "query_ledger" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    fx = load_fx_rates()

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)

    # Real dataset has a documented FX gap for Q3 2024 -> this is one of the
    # two clean refusals per the master prompt.
    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.result is None
    assert len(bundle.calculations) >= 2
