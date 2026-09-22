"""Tests for finance_assistant.workflows.travel.travel_comparison (Q2).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected totals/bridges are hand-computed, never
derived by calling the function under test.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_chart_of_accounts, load_fx_rates, load_gl_transactions
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.workflows.travel import travel_comparison


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-03-05",
        "accrual_date": "2024-03-05",
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "TE1",
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
        "account_code": "TE1",
        "account_name": "Client Entertainment",
        "parent_code": "P1",
        "parent_name": "Travel & Entertainment",
        "statement_line": "Operating Expenses",
        "valid_from": "2023-01-01",
        "valid_to": "9999-12-31",
    }
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-03", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, coa_rows, fx_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    coa_path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], coa_rows)
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows)
    return load_gl_transactions(gl_path), load_chart_of_accounts(coa_path), load_fx_rates(fx_path)


def test_reclassification_forces_partial_via_grouping_fact(write_csv):
    # A reclassification between the two years makes grouping_would_change_result
    # genuinely True (bridge_usd != 0) -> PARTIAL, not hardcoded.
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-08-05", accrual_date="2024-08-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-08-05", accrual_date="2023-08-05", amount="900.00"),
        ],
        [
            _coa_row(valid_from="2023-01-01", valid_to="2024-06-30", parent_name="Travel & Entertainment"),
            _coa_row(valid_from="2024-07-01", valid_to="9999-12-31", parent_name="Client Relations"),
        ],
        [_fx_row(period_month="2024-08"), _fx_row(period_month="2023-08")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.status == AnswerStatus.PARTIAL


def test_grouping_fact_is_computed_not_hardcoded_when_no_reclassification(write_csv):
    # No reclassification anywhere in the COA's date range and full FX
    # coverage -> grouping_would_change_result is genuinely False, proving
    # PARTIAL is not asserted unconditionally by this workflow.
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-03-05", accrual_date="2024-03-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-03-05", accrual_date="2023-03-05", amount="900.00"),
        ],
        [_coa_row(valid_from="2023-01-01", valid_to="9999-12-31")],
        [_fx_row(period_month="2024-03"), _fx_row(period_month="2023-03")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result["reclassification_bridge_usd"]["current"]["current"] == pytest.approx(0.0)


def test_reclassification_produces_both_comparable_bases_and_nonzero_bridge(write_csv):
    # TE1 is "Travel & Entertainment" only up to 2024-06-30, then reclassified
    # to a different parent from 2024-07-01 onward — a mid-year change in the
    # COA's own validity window, not a change in account_code.
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-08-05", accrual_date="2024-08-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-08-05", accrual_date="2023-08-05", amount="900.00"),
        ],
        [
            _coa_row(valid_from="2023-01-01", valid_to="2024-06-30", parent_name="Travel & Entertainment"),
            _coa_row(valid_from="2024-07-01", valid_to="9999-12-31", parent_name="Client Relations"),
        ],
        [_fx_row(period_month="2024-08"), _fx_row(period_month="2023-08")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    # Reported basis: FY2024's row (dated 2024-08) resolves against the
    # post-reclassification COA row -> parent is "Client Relations", so it is
    # NOT counted in reported T&E. FY2023's row resolves against the
    # pre-reclassification row -> parent IS "Travel & Entertainment".
    assert bundle.result["reported_basis"]["current"]["te_total_usd"] == pytest.approx(0.0)
    assert bundle.result["reported_basis"]["prior"]["te_total_usd"] == pytest.approx(900.0)

    # Comparable basis "current" (reference = max date = 2024-08-05, the
    # post-reclassification row applies to BOTH years) -> neither year's row
    # is T&E under that classification.
    assert bundle.result["comparable_basis_current"]["current"]["te_total_usd"] == pytest.approx(0.0)
    assert bundle.result["comparable_basis_current"]["prior"]["te_total_usd"] == pytest.approx(0.0)

    # Comparable basis "prior" (reference = min date = 2023-08-05, the
    # pre-reclassification row applies to BOTH years) -> both rows count.
    assert bundle.result["comparable_basis_prior"]["current"]["te_total_usd"] == pytest.approx(1000.0)
    assert bundle.result["comparable_basis_prior"]["prior"]["te_total_usd"] == pytest.approx(900.0)

    assert bundle.result["reclassification_bridge_usd"]["prior"]["current"] == pytest.approx(1000.0)


def test_sign_mismatch_between_reported_and_comparable_triggers_warning(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-08-05", accrual_date="2024-08-05", amount="100.00"),
            _gl_row(txn_id="T2", posting_date="2023-08-05", accrual_date="2023-08-05", amount="900.00"),
        ],
        [
            _coa_row(valid_from="2023-01-01", valid_to="2024-06-30", parent_name="Travel & Entertainment"),
            _coa_row(valid_from="2024-07-01", valid_to="9999-12-31", parent_name="Client Relations"),
        ],
        [_fx_row(period_month="2024-08"), _fx_row(period_month="2023-08")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    # Reported: FY2024's row resolves against the post-reclassification COA
    # row (parent "Client Relations") -> excluded, reported current = 0.
    # FY2023's row resolves against the pre-reclassification row (parent
    # "Travel & Entertainment") -> included, reported prior = 900.
    # variance = 0 - 900 = -900 (down).
    assert "comparable_basis_current" in bundle.result
    assert "comparable_basis_prior" in bundle.result
    assert bundle.result["reported_basis"]["variance_usd"] == pytest.approx(-900.0)
    # comparable "prior" basis: both rows classified under the
    # pre-reclassification parent -> current(100) - prior(900) = -800, same
    # sign as reported (-900) -- no mismatch on this basis.
    assert bundle.result["comparable_basis_prior"]["variance_usd"] == pytest.approx(-800.0)


def test_no_reclassification_between_years_still_reports_both_bases(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-03-05", accrual_date="2024-03-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-03-05", accrual_date="2023-03-05", amount="900.00"),
        ],
        [_coa_row(valid_from="2023-01-01", valid_to="9999-12-31")],
        [_fx_row(period_month="2024-03"), _fx_row(period_month="2023-03")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.result["reclassification_bridge_usd"]["current"]["current"] == pytest.approx(0.0)
    assert bundle.result["reclassification_bridge_usd"]["prior"]["current"] == pytest.approx(0.0)
    assert bundle.result["reported_basis"]["current"]["te_total_usd"] == pytest.approx(1000.0)
    assert bundle.result["comparable_basis_current"]["current"]["te_total_usd"] == pytest.approx(1000.0)


def test_missing_fx_rate_reflected_in_coverage(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-03-05", accrual_date="2024-03-05", currency="EUR", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-03-05", accrual_date="2023-03-05", amount="900.00"),
        ],
        [_coa_row(valid_from="2023-01-01", valid_to="9999-12-31")],
        [_fx_row(period_month="2023-03")],  # no 2024-03 EUR rate
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.coverage.computable_amount_pct < 100.0


def test_tool_calls_are_recorded(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", posting_date="2024-03-05", accrual_date="2024-03-05", amount="1000.00"),
            _gl_row(txn_id="T2", posting_date="2023-03-05", accrual_date="2023-03-05", amount="900.00"),
        ],
        [_coa_row(valid_from="2023-01-01", valid_to="9999-12-31")],
        [_fx_row(period_month="2024-03"), _fx_row(period_month="2023-03")],
    )

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {
        "load_policy_rules",
        "query_ledger",
        "resolve_account_hierarchy",
        "convert_to_usd",
        "aggregate_usd",
        "search_documents",
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

    bundle = travel_comparison(gl, coa, fx, year_current=2024, year_prior=2023)

    assert bundle.status == AnswerStatus.PARTIAL
    assert bundle.result is not None
    assert "reported_basis" in bundle.result
    assert "comparable_basis_current" in bundle.result
    assert "comparable_basis_prior" in bundle.result
