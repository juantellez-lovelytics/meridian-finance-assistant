"""Tests for finance_assistant.evidence.render.render_bundle_text.

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv or data/documents/*.md content. These assert the renderer
faithfully surfaces bundle fields as text (what the forbidden_claims eval
check runs its regexes against), not any particular dataset's numbers.
"""

from finance_assistant import config
from finance_assistant.data.loaders import load_fx_rates, load_gl_transactions
from finance_assistant.evidence.render import render_bundle_text
from finance_assistant.workflows.consolidated import consolidated_spend
from finance_assistant.workflows.headcount import headcount_cost_per_fte


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


def test_refused_bundle_renders_reason_and_reason_code_never_a_result(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHeadcount is tracked by HR, not finance.\n")

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)
    text = render_bundle_text(bundle)

    assert "status: refused" in text
    assert bundle.refusal_reason in text
    assert "missing_fte_denominator" in text
    assert "result:" not in text


def test_refused_with_calculations_still_renders_computable_components(write_csv):
    gl_path = write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [
            _gl_row(txn_id="T1", entity="E1", currency="USD", amount="1000.00"),
            _gl_row(txn_id="T2", entity="E2", currency="EUR", amount="500.00"),
        ],
    )
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], [_fx_row(currency="USD")])
    gl = load_gl_transactions(gl_path)
    fx = load_fx_rates(fx_path)

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)
    text = render_bundle_text(bundle)

    assert "status: refused" in text
    assert "calculations:" in text
    assert "computable USD total (FX-convertible rows only)" in text
    assert "computable components by entity (USD)" in text
    assert "non-convertible local amount by currency" in text


def test_answer_bundle_renders_result_contents(write_csv):
    gl_path = write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [_gl_row(txn_id="T1", entity="E1", amount="1500.00")],
    )
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], [_fx_row()])
    gl = load_gl_transactions(gl_path)
    fx = load_fx_rates(fx_path)

    bundle = consolidated_spend(gl, fx, "Q3", year=2024)
    text = render_bundle_text(bundle)

    assert "status: answer" in text
    assert "result:" in text
    assert "exact_total_usd" in text
    assert "refusal_reason" not in text
