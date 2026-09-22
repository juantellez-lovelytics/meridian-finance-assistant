"""Tests for finance_assistant.workflows.policy.te_policy_check (Q6).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected finding counts are hand-computed, never
derived by calling the function under test.
"""

from finance_assistant import config
from finance_assistant.data.loaders import load_chart_of_accounts, load_fx_rates, load_gl_transactions
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.workflows.policy import te_policy_check


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-04-05",
        "accrual_date": "2024-04-05",
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "amount": "1500.00",
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
        "account_name": "Some Account",
        "parent_code": "P1",
        "parent_name": "Travel & Entertainment",
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


def test_confirmed_and_insufficient_findings_still_answer(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [
            # No approval_ref, amount >= pre_approval threshold (1000) -> CONFIRMED_RULE_MATCH.
            _gl_row(txn_id="T1", amount="1500.00", approval_ref=""),
        ],
        [_coa_row()],
        [_fx_row()],
    )

    bundle = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31")

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result["findings_by_rule"]["pre_approval_threshold"]["CONFIRMED_RULE_MATCH"] == 1


def test_unmapped_account_warns_but_still_answers(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", account_code="UNKNOWN_CODE")],
        [_coa_row(account_code="AC1")],  # AC1 defined, UNKNOWN_CODE is not
        [_fx_row()],
    )

    bundle = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31")

    assert bundle.status == AnswerStatus.ANSWER
    assert any("unmapped" in w for w in bundle.warnings)


def test_reported_vs_policy_perimeter_basis_differ_row_counts(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", account_code="AC1", posting_date="2024-08-05", accrual_date="2024-08-05")],
        [
            _coa_row(account_code="AC1", valid_from="2023-01-01", valid_to="2024-06-30", parent_name="Travel & Entertainment"),
            _coa_row(account_code="AC1", valid_from="2024-07-01", valid_to="9999-12-31", parent_name="Client Relations"),
        ],
        [_fx_row(period_month="2024-08")],
    )

    reported = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31", perimeter_basis="reported")
    policy_basis = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31", perimeter_basis="policy")

    assert reported.result["perimeter_rows"] != policy_basis.result["perimeter_rows"]


def test_sources_include_a_source_ref_per_cited_section(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", amount="1500.00", approval_ref="")],
        [_coa_row()],
        [_fx_row()],
    )

    bundle = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31")

    assert len(bundle.sources) >= 1
    assert all(s.evidence_id for s in bundle.sources)


def test_tool_calls_are_recorded(write_csv):
    gl, coa, fx = _load(
        write_csv,
        [_gl_row(txn_id="T1", amount="1500.00", approval_ref="")],
        [_coa_row()],
        [_fx_row()],
    )

    bundle = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31")

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {"query_ledger", "load_policy_rules", "evaluate_te_policy", "search_documents"}
    assert "query_ledger" in tool_names
    assert "evaluate_te_policy" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()
    fx = load_fx_rates()

    bundle = te_policy_check(gl, coa, fx, "2024-01-01", "2024-12-31")

    assert bundle.status == AnswerStatus.ANSWER
    assert bundle.result is not None
    assert sum(bundle.result["findings_by_state"].values()) > 0
