"""Tests for finance_assistant.cli.main.

All I/O happens under tmp_path -- a synthetic dataset built via the
write_csv/write_markdown fixtures, never the real data/*.csv files -- and
the CLI runs in-process via main(argv), matching evals/run_evals.py's own
testable main() -> int convention.
"""

import json

from finance_assistant import config
from finance_assistant.cli import main


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


def _budget_row(**overrides):
    row = {
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "period_month": "2024-04",
        "budget_amount": "500.00",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def _vendor_row(**overrides):
    row = {"vendor_id": "V1", "vendor_name": "Acme Inc", "category": "Services", "country": "US"}
    row.update(overrides)
    return row


def _write_dataset(write_csv, write_markdown, gl_rows=None, coa_rows=None, fx_rows=None, budget_rows=None, vendor_rows=None):
    write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows if gl_rows is not None else [_gl_row()])
    write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], coa_rows if coa_rows is not None else [_coa_row()])
    write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows if fx_rows is not None else [_fx_row()])
    write_csv(
        config.BUDGET_SCHEMA["filename"], config.BUDGET_SCHEMA["required_columns"], budget_rows if budget_rows is not None else [_budget_row()]
    )
    write_csv(
        config.VENDORS_SCHEMA["filename"], config.VENDORS_SCHEMA["required_columns"], vendor_rows if vendor_rows is not None else [_vendor_row()]
    )
    write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHeadcount is tracked by HR, not finance.\n")


def _write_question(tmp_path, filename, intent, params, question="test question"):
    path = tmp_path / filename
    path.write_text(json.dumps({"intent": intent, "question": question, "params": params}), encoding="utf-8")
    return path


def test_cli_runs_opex_and_writes_trace_and_prints_answer(tmp_path, write_csv, write_markdown, capsys):
    _write_dataset(write_csv, write_markdown)
    question_file = _write_question(tmp_path, "q.json", "opex_by_cost_centre", {"quarter": "Q2", "year": 2024})
    traces_dir = tmp_path / "traces"

    exit_code = main([str(question_file), "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])

    assert exit_code == 0
    out = capsys.readouterr()
    assert "status: answer" in out.out

    trace_files = list(traces_dir.glob("*.json"))
    assert len(trace_files) == 1
    trace_data = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_data["status"] == "answer"
    assert trace_data["steps"]


def test_cli_refused_status_still_exits_zero(tmp_path, write_csv, write_markdown, capsys):
    _write_dataset(
        write_csv,
        write_markdown,
        gl_rows=[
            _gl_row(txn_id="T1", entity="E1", currency="USD", amount="1000.00"),
            _gl_row(txn_id="T2", entity="E2", currency="EUR", amount="500.00"),
        ],
        fx_rows=[_fx_row(currency="USD")],  # no EUR rate
    )
    question_file = _write_question(tmp_path, "q.json", "consolidated_spend", {"quarter": "Q2", "year": 2024})
    traces_dir = tmp_path / "traces"

    exit_code = main([str(question_file), "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])

    assert exit_code == 0
    out = capsys.readouterr()
    assert "status: refused" in out.out

    trace_data = json.loads(next(traces_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert trace_data["status"] == "refused"


def test_cli_missing_question_file_errors_clearly(tmp_path, capsys):
    exit_code = main([str(tmp_path / "does_not_exist.json")])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_malformed_json_errors_clearly(tmp_path, capsys):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json", encoding="utf-8")

    exit_code = main([str(bad_file)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "JSON" in err


def test_cli_unknown_intent_lists_valid_options(tmp_path, capsys):
    question_file = _write_question(tmp_path, "q.json", "not_a_real_intent", {})

    exit_code = main([str(question_file)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "unknown intent" in err
    assert "opex_by_cost_centre" in err


def test_cli_missing_required_param_errors_clearly(tmp_path, capsys):
    question_file = _write_question(tmp_path, "q.json", "opex_by_cost_centre", {})

    exit_code = main([str(question_file)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "quarter" in err


def test_cli_rejects_dataframe_key_in_params(tmp_path, capsys):
    question_file = _write_question(tmp_path, "q.json", "opex_by_cost_centre", {"quarter": "Q2", "gl": {}})

    exit_code = main([str(question_file)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "gl" in err
    assert "dataframe" in err.lower()


def test_cli_two_runs_do_not_overwrite_each_others_trace(tmp_path, write_csv, write_markdown, capsys):
    _write_dataset(write_csv, write_markdown)
    question_file = _write_question(tmp_path, "q.json", "opex_by_cost_centre", {"quarter": "Q2", "year": 2024})
    traces_dir = tmp_path / "traces"

    main([str(question_file), "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])
    main([str(question_file), "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])

    assert len(list(traces_dir.glob("*.json"))) == 2


def test_cli_free_text_mode_uses_keyword_fallback_without_credential(tmp_path, write_csv, write_markdown, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _write_dataset(write_csv, write_markdown)
    traces_dir = tmp_path / "traces"

    exit_code = main(["What was our opex by cost centre in Q2?", "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "intent: opex_by_cost_centre" in out
    assert "keyword fallback" in out

    trace_data = json.loads(next(traces_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert trace_data["model_calls"] == []


def test_cli_free_text_containing_json_substring_is_not_misrouted_to_json_mode(tmp_path, write_csv, write_markdown, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_dataset(write_csv, write_markdown)
    traces_dir = tmp_path / "traces"

    # Contains the substring "json" but does not end in ".json" -- must
    # still be routed to free-text mode, not misread as a JSON file path.
    exit_code = main(["is our json export of opex by cost centre correct?", "--data-dir", str(tmp_path), "--traces-dir", str(traces_dir)])

    assert exit_code in (0, 1)
    assert len(list(traces_dir.glob("*.json"))) == 1
