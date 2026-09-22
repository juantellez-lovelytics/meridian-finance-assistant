"""Tests for finance_assistant.workflows.opex.opex_by_cost_centre (Q1).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected totals are hand-computed, never derived by
calling the function under test.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_chart_of_accounts, load_fx_rates, load_gl_transactions
from finance_assistant.evidence.models import AnswerStatus, MissingEvidenceReasonCode
from finance_assistant.workflows.opex import opex_by_cost_centre


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-04-05",
        "accrual_date": "2024-04-05",
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


def _coa_row(**overrides):
    row = {
        "account_code": "AC1",
        "account_name": "Some Opex Account",
        "parent_code": "P1",
        "parent_name": "Operations",
        "statement_line": "Operating Expenses",
        "valid_from": "2023-01-01",
        "valid_to": "9999-12-31",
    }
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-04", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, coa_rows, fx_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    coa_path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], coa_rows)
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows)
    return load_gl_transactions(gl_path), load_chart_of_accounts(coa_path), load_fx_rates(fx_path)


def test_explicit_year_with_full_fx_coverage(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", cost_centre="CC-A", amount="1000.00"),
            _gl_row(txn_id="T2", cost_centre="CC-B", amount="500.00"),
        ],
        [_coa_row()],
        [_fx_row()],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result["total_usd"] == pytest.approx(1500.0)
    assert bundle.result["by_cost_centre_usd"]["CC-A"] == pytest.approx(1000.0)
    assert bundle.result["by_cost_centre_usd"]["CC-B"] == pytest.approx(500.0)


def test_year_none_with_materially_different_years_needs_clarification(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2023-04-05", accrual_date="2023-04-05", amount="100.00"),
            _gl_row(txn_id="T2", posting_date="2024-04-05", accrual_date="2024-04-05", amount="10000.00"),
        ],
        [_coa_row()],
        [_fx_row(period_month="2023-04"), _fx_row(period_month="2024-04")],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=None)

    assert bundle.status == AnswerStatus.NEEDS_CLARIFICATION
    assert set(bundle.clarification_options) == {"FY2023 Q2", "FY2024 Q2"}
    assert bundle.result is None
    # No year was actually defaulted here -- the gate rejected the default,
    # so the bundle must not claim it applied one.
    assert not any("defaulted" in a for a in bundle.assumptions)


def test_year_none_with_close_years_answers_with_most_recent(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2023-04-05", accrual_date="2023-04-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2024-04-05", accrual_date="2024-04-05", amount="1010.00"),
        ],
        [_coa_row()],
        [_fx_row(period_month="2023-04"), _fx_row(period_month="2024-04")],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=None)

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result["year"] == 2024
    assert any("defaulted" in a for a in bundle.assumptions)


def test_missing_fx_rate_surfaces_as_missing_evidence(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", currency="EUR", amount="1000.00")],
        [_coa_row()],
        [],  # no EUR rate at all
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.status != AnswerStatus.ANSWER
    assert len(bundle.missing_evidence) == 1
    assert bundle.missing_evidence[0].reason_code == MissingEvidenceReasonCode.MISSING_FX_RATE


def test_perimeter_filter_excludes_non_opex_accounts(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", account_code="AC1", amount="1000.00"),
            _gl_row(txn_id="T2", account_code="AC2", amount="9999.00"),
        ],
        [_coa_row(account_code="AC1"), _coa_row(account_code="AC2", statement_line="Cost of Goods Sold")],
        [_fx_row()],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.result["opex_perimeter_rows"] < bundle.result["total_rows_before_perimeter_filter"]
    assert bundle.result["total_usd"] == pytest.approx(1000.0)


def test_perimeter_filter_excludes_nothing_when_all_accounts_share_statement_line(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", account_code="AC1", amount="1000.00")],
        [_coa_row(account_code="AC1")],
        [_fx_row()],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.result["opex_perimeter_rows"] == bundle.result["total_rows_before_perimeter_filter"]
    assert any("did not exclude any row" in a for a in bundle.assumptions)


def test_tool_calls_are_recorded(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", amount="1000.00")],
        [_coa_row()],
        [_fx_row()],
    )

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {
        "query_ledger",
        "resolve_account_hierarchy",
        "normalize_reporting_cost_centre",
        "convert_to_usd",
        "aggregate_usd",
        "aggregate_usd_by",
    }
    assert "query_ledger" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()
    fx = load_fx_rates()

    bundle = opex_by_cost_centre(gl, coa, fx, "Q2", year=2024)

    assert bundle.status in (AnswerStatus.ANSWER, AnswerStatus.PARTIAL)
    assert bundle.result is not None
    assert bundle.result["total_usd"] >= 0
