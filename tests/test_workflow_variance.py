"""Tests for finance_assistant.workflows.variance.budget_variance (Q5).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected variances are hand-computed, never derived
by calling the function under test.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import (
    load_budget,
    load_chart_of_accounts,
    load_fx_rates,
    load_gl_transactions,
)
from finance_assistant.evidence.models import AnswerStatus, MissingEvidenceReasonCode
from finance_assistant.workflows.variance import budget_variance


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


def _coa_row(**overrides):
    row = {
        "account_code": "AC1",
        "account_name": "Freight Charges",
        "parent_code": "P1",
        "parent_name": "Operations",
        "statement_line": "Operating Expenses",
        "valid_from": "2023-01-01",
        "valid_to": "9999-12-31",
    }
    row.update(overrides)
    return row


def _budget_row(**overrides):
    row = {
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "period_month": "2024-07",
        "budget_amount": "500.00",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-07", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, coa_rows, fx_rows, budget_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    coa_path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], coa_rows)
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows)
    budget_path = write_csv(config.BUDGET_SCHEMA["filename"], config.BUDGET_SCHEMA["required_columns"], budget_rows)
    return (
        load_gl_transactions(gl_path),
        load_chart_of_accounts(coa_path),
        load_fx_rates(fx_path),
        load_budget(budget_path),
    )


def test_worst_centre_and_driver_account_match_hand_computed_values(write_csv):
    gl, coa, fx, budget = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00"),
            _gl_row(txn_id="T2", cost_centre="CC-B", account_code="AC1", amount="100.00"),
        ],
        [_coa_row()],
        [_fx_row()],
        [
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500.00"),
            _budget_row(cost_centre="CC-B", account_code="AC1", budget_amount="200.00"),
        ],
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024)

    # Full FX coverage, no duplicate budget keys, full account mapping ->
    # nothing degrades the gate's draft ANSWER; PARTIAL in the real dataset
    # comes from genuine FX/duplicate-key gaps, not an asserted constant.
    assert bundle.status == AnswerStatus.ANSWER
    # CC-A variance = 1000 - 500 = 500 (most adverse); CC-B = 100 - 200 = -100.
    assert bundle.result["worst_cost_centre"] == "CC-A"
    assert bundle.result["driver_account"]["account_code"] == "AC1"
    assert bundle.result["driver_account"]["variance_usd"] == pytest.approx(500.0)


def test_memo_discrepancy_warning_when_narrative_does_not_match_driver(write_csv):
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00")],
        [_coa_row(account_name="Office Supplies")],  # not discussed anywhere in board_memo_2024_q2.md
        [_fx_row()],
        [_budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500.00")],
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024)

    assert any("Office Supplies" in w for w in bundle.warnings)
    assert bundle.sources == []
    assert bundle.result["driver_account"]["memo_confirmation"] == "NO_MATCH"


def test_partial_term_overlap_is_not_treated_as_confirmation(write_csv, write_markdown):
    # The memo discusses "expedited freight" specifically, but the real
    # driver account is "outbound freight" -- a different account that
    # merely shares the word "freight". Term overlap on one shared word
    # must never read as the memo confirming this specific account.
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00")],
        [_coa_row(account_name="Outbound Freight")],
        [_fx_row()],
        [_budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500.00")],
    )
    documents_dir = write_markdown(
        "board_memo_2024_q2.md",
        "## Freight\n\nExpedited freight has run ahead of plan following a tooling failure.\n",
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024, documents_dir=documents_dir)

    assert bundle.result["driver_account"]["memo_confirmation"] == "PARTIAL_OVERLAP"
    assert bundle.sources == []
    assert any(
        "partially overlaps" in w and "Outbound Freight" in w and "outbound" in w for w in bundle.warnings
    )


def test_full_term_coverage_is_confirmed(write_csv, write_markdown):
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00")],
        [_coa_row(account_name="Outbound Freight")],
        [_fx_row()],
        [_budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500.00")],
    )
    documents_dir = write_markdown(
        "board_memo_2024_q2.md",
        "## Freight\n\nOutbound freight has run ahead of plan this quarter.\n",
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024, documents_dir=documents_dir)

    assert bundle.result["driver_account"]["memo_confirmation"] == "CONFIRMED"
    assert len(bundle.sources) == 1


def test_year_outside_budget_coverage_refuses(write_csv):
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(posting_date="2099-07-05", accrual_date="2099-07-05")],
        [_coa_row()],
        [_fx_row(period_month="2099-07")],
        [_budget_row(period_month="2024-07")],
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2099)

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.result is None
    assert bundle.missing_evidence[0].reason_code == MissingEvidenceReasonCode.NO_DATA_FOR_PERIOD


def test_duplicate_budget_keys_reported_as_diagnostic(write_csv):
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00")],
        [_coa_row()],
        [_fx_row()],
        [
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="300.00"),
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="200.00"),
        ],
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024)

    assert bundle.result["duplicate_budget_keys"] == 1
    assert any("duplicate budget key" in w for w in bundle.warnings)


def test_tool_calls_are_recorded(write_csv, write_markdown):
    gl, coa, fx, budget = _load(
        write_csv,
        [_gl_row(txn_id="T1", cost_centre="CC-A", account_code="AC1", amount="1000.00")],
        [_coa_row(account_name="Outbound Freight")],
        [_fx_row()],
        [_budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500.00")],
    )
    documents_dir = write_markdown(
        "board_memo_2024_q2.md",
        "## Freight\n\nOutbound freight has run ahead of plan this quarter.\n",
    )

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024, documents_dir=documents_dir)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {
        "query_ledger",
        "resolve_account_hierarchy",
        "normalize_reporting_cost_centre",
        "convert_to_usd",
        "aggregate_usd_by",
        "query_budget",
        "search_documents",
    }
    assert "query_ledger" in tool_names
    assert "query_budget" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()
    fx = load_fx_rates()
    budget = load_budget()

    bundle = budget_variance(gl, coa, fx, budget, "Q3", 2024)

    assert bundle.status == AnswerStatus.PARTIAL
    assert bundle.result is not None
    assert bundle.result["worst_cost_centre"] is not None
    assert bundle.result["driver_account"] is not None
