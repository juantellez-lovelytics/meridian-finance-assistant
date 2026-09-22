"""Tests for finance_assistant.orchestration.orchestrator.answer_question.

All datasets are synthetic, built via the write_csv/write_markdown
fixtures under tmp_path -- never the real data/*.csv files.
"""

from pathlib import Path

import pytest
import yaml

from finance_assistant import config
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.evidence.trace import ModelCall
from finance_assistant.orchestration.interpreter import INTENT_DESCRIPTIONS, NEARBY_INTENTS
from finance_assistant.orchestration.intents import Intent, IntentRequest
from finance_assistant.orchestration.orchestrator import answer_question
from finance_assistant.orchestration.settings import load_settings

QUESTIONS_YAML = yaml.safe_load((Path(__file__).parent.parent / "evals" / "questions.yaml").read_text(encoding="utf-8"))
_ALL_EIGHT_CASES = [(case["question"], Intent[case["expected_intent"]]) for case in QUESTIONS_YAML["cases"]]


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-03-05",
        "accrual_date": "2024-03-05",
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "OP1",
        "amount": "500.00",
        "currency": "USD",
        "vendor_id": "V1",
        "doc_ref": "D1",
        "approval_ref": "",
        "memo": "",
    }
    row.update(overrides)
    return row


def _coa_row(**overrides):
    row = {
        "account_code": "OP1",
        "account_name": "Office Supplies",
        "parent_code": "P1",
        "parent_name": "Operations",
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


def _budget_row(**overrides):
    row = {
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "OP1",
        "period_month": "2024-07",
        "budget_amount": "100.00",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def _vendor_row(**overrides):
    row = {"vendor_id": "V1", "vendor_name": "Vendor One", "category": "Services", "country": "US"}
    row.update(overrides)
    return row


def _write_full_dataset(write_csv, write_markdown) -> None:
    """A dataset that lets all 8 workflows run to completion (any legitimate
    status -- ANSWER/PARTIAL/REFUSED/NEEDS_CLARIFICATION -- never a crash),
    covering: opex + travel accounts across two years, a Q3-2024 budget,
    two vendors, and a documents/ dir (board_memo, needed by
    budget_variance's driver search)."""
    write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [
            _gl_row(txn_id="T1", posting_date="2023-03-05", accrual_date="2023-03-05", account_code="OP1", amount="500.00", vendor_id="V1"),
            _gl_row(txn_id="T2", posting_date="2023-08-05", accrual_date="2023-08-05", account_code="TE1", amount="300.00", vendor_id="V2"),
            _gl_row(txn_id="T3", posting_date="2024-03-05", accrual_date="2024-03-05", account_code="OP1", amount="600.00", vendor_id="V1"),
            _gl_row(txn_id="T4", posting_date="2024-08-05", accrual_date="2024-08-05", account_code="TE1", amount="350.00", vendor_id="V2"),
            _gl_row(
                txn_id="T5",
                posting_date="2024-08-06",
                accrual_date="2024-08-06",
                cost_centre="CC-B",
                account_code="OP1",
                amount="200.00",
                vendor_id="V1",
            ),
            _gl_row(txn_id="T6", posting_date="2024-09-10", accrual_date="2024-09-10", account_code="OP1", amount="250.00", vendor_id="V1"),
        ],
    )
    write_csv(
        config.COA_SCHEMA["filename"],
        config.COA_SCHEMA["required_columns"],
        [
            _coa_row(account_code="OP1", account_name="Office Supplies", parent_name="Operations"),
            _coa_row(account_code="TE1", account_name="Client Entertainment", parent_code="P2", parent_name="Travel & Entertainment"),
        ],
    )
    write_csv(
        config.FX_SCHEMA["filename"],
        config.FX_SCHEMA["required_columns"],
        [_fx_row(period_month=month) for month in ("2023-03", "2023-08", "2024-03", "2024-07", "2024-08", "2024-09")],
    )
    write_csv(
        config.BUDGET_SCHEMA["filename"],
        config.BUDGET_SCHEMA["required_columns"],
        [
            _budget_row(cost_centre=cc, period_month=month, budget_amount=amount)
            for cc, amount in (("CC-A", "100.00"), ("CC-B", "150.00"))
            for month in ("2024-07", "2024-08", "2024-09")
        ],
    )
    write_csv(
        config.VENDORS_SCHEMA["filename"],
        config.VENDORS_SCHEMA["required_columns"],
        [_vendor_row(vendor_id="V1", vendor_name="Vendor One"), _vendor_row(vendor_id="V2", vendor_name="Vendor Two", category="Travel")],
    )
    write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHeadcount is tracked by HR, not finance.\n")


def _write_travel_dataset(write_csv, years: list[int]) -> None:
    """gl/coa/fx covering a Travel & Entertainment account in each of
    `years`, with no reclassification -- used by the dataset-defaulting
    tests below, which only exercise travel_comparison."""
    gl_rows = [
        _gl_row(txn_id=f"T{y}", posting_date=f"{y}-06-15", accrual_date=f"{y}-06-15", account_code="TE1", amount="300.00", vendor_id="V2")
        for y in years
    ]
    write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    write_csv(
        config.COA_SCHEMA["filename"],
        config.COA_SCHEMA["required_columns"],
        [_coa_row(account_code="TE1", account_name="Client Entertainment", parent_name="Travel & Entertainment")],
    )
    write_csv(
        config.FX_SCHEMA["filename"],
        config.FX_SCHEMA["required_columns"],
        [_fx_row(period_month=f"{y}-06") for y in years],
    )


def test_no_credential_falls_back_to_keyword_interpreter_and_answers_all_eight_questions(
    tmp_path, write_csv, write_markdown, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _write_full_dataset(write_csv, write_markdown)
    settings = load_settings()
    assert not settings.has_credential()

    for question, expected_intent in _ALL_EIGHT_CASES:
        bundle, trace = answer_question(question, data_dir=tmp_path, documents_dir=tmp_path / "documents", settings=settings)

        assert bundle.status is not AnswerStatus.ERROR, f"{question!r} -> ERROR: {bundle.warnings}"
        assert bundle.intent == expected_intent, f"{question!r} -> {bundle.intent}, expected {expected_intent}"
        assert any("keyword" in a for a in bundle.assumptions), f"{question!r}: {bundle.assumptions}"
        assert trace.model_calls == []


def test_low_confidence_returns_needs_clarification_without_running_workflow(tmp_path, fake_llm_client):
    client = fake_llm_client(IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.1))
    settings = load_settings(env={"LLM_MODEL": "anthropic/x", "LLM_MIN_CONFIDENCE": "0.5"})

    bundle, _trace = answer_question("some opex question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status == AnswerStatus.NEEDS_CLARIFICATION
    assert bundle.result is None
    expected_options = [
        INTENT_DESCRIPTIONS[Intent.OPEX_BY_COST_CENTRE],
        *(INTENT_DESCRIPTIONS[i] for i in NEARBY_INTENTS[Intent.OPEX_BY_COST_CENTRE]),
    ]
    assert bundle.clarification_options == expected_options
    assert Intent.OPEX_BY_COST_CENTRE.value not in bundle.clarification_options


def test_unknown_intent_returns_needs_clarification_with_full_catalog(tmp_path, fake_llm_client):
    client = fake_llm_client(IntentRequest(intent=Intent.UNKNOWN, confidence=0.0))
    settings = load_settings(env={"LLM_MODEL": "anthropic/x"})

    bundle, _trace = answer_question("what's the weather like", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status == AnswerStatus.NEEDS_CLARIFICATION
    assert bundle.result is None
    expected_options = [INTENT_DESCRIPTIONS[i] for i in Intent if i is not Intent.UNKNOWN]
    assert bundle.clarification_options == expected_options
    assert len(bundle.clarification_options) == 8


def test_call_count_ceiling_exceeded_returns_error(tmp_path, fake_llm_client):
    client = fake_llm_client(IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.9))
    settings = load_settings(
        env={"LLM_MODEL": "anthropic/x", "LLM_MAX_CALLS_PER_QUESTION": "0", "LLM_MAX_COST_USD_PER_QUESTION": "999", "LLM_MIN_CONFIDENCE": "0.0"}
    )

    bundle, _trace = answer_question("some question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status == AnswerStatus.ERROR
    assert any("ceiling" in w for w in bundle.warnings)
    assert any("call" in w for w in bundle.warnings)


def test_token_ceiling_exceeded_returns_error(tmp_path, fake_llm_client):
    client = fake_llm_client(
        IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.9), prompt_tokens=1000, completion_tokens=500
    )
    settings = load_settings(
        env={"LLM_MODEL": "anthropic/x", "LLM_MAX_TOKENS_PER_QUESTION": "100", "LLM_MIN_CONFIDENCE": "0.0"}
    )

    bundle, _trace = answer_question("some question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status == AnswerStatus.ERROR
    assert any("ceiling" in w for w in bundle.warnings)
    assert any("token" in w for w in bundle.warnings)


def test_cost_ceiling_exceeded_returns_error(tmp_path, fake_llm_client):
    client = fake_llm_client(IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.9), estimated_cost_usd=10.0)
    settings = load_settings(env={"LLM_MODEL": "anthropic/x", "LLM_MAX_COST_USD_PER_QUESTION": "0.01", "LLM_MIN_CONFIDENCE": "0.0"})

    bundle, _trace = answer_question("some question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status == AnswerStatus.ERROR
    assert any("cost" in w for w in bundle.warnings)


def test_unknown_cost_does_not_trip_cost_ceiling_but_is_declared(tmp_path, fake_llm_client):
    # Low confidence deliberately -- so the run resolves via the
    # NEEDS_CLARIFICATION short-circuit and never needs a real dataset,
    # while still exercising the ceiling check's "unknown" branch first.
    client = fake_llm_client(IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.1), estimated_cost_usd="unknown")
    settings = load_settings(env={"LLM_MODEL": "anthropic/x", "LLM_MAX_COST_USD_PER_QUESTION": "0.01", "LLM_MIN_CONFIDENCE": "0.5"})

    bundle, _trace = answer_question("some question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert bundle.status != AnswerStatus.ERROR
    assert any("pricing unknown" in a for a in bundle.assumptions)


def test_dataset_derived_defaults_fill_params_the_llm_could_not_supply(tmp_path, write_csv, fake_llm_client):
    _write_travel_dataset(write_csv, years=[2023, 2024])
    client = fake_llm_client(IntentRequest(intent=Intent.TRAVEL_COMPARISON, confidence=0.9))
    settings = load_settings(env={"LLM_MODEL": "anthropic/x"})

    bundle, _trace = answer_question(
        "how does travel spend compare year over year", data_dir=tmp_path, documents_dir=tmp_path / "documents", llm_client=client, settings=settings
    )

    assert bundle.status is not AnswerStatus.ERROR
    assert bundle.result is not None
    assert bundle.result["years"] == {"current": 2024, "prior": 2023}


def test_explicit_llm_params_override_dataset_derived_defaults(tmp_path, write_csv, fake_llm_client):
    _write_travel_dataset(write_csv, years=[2022, 2023, 2024])
    client = fake_llm_client(IntentRequest(intent=Intent.TRAVEL_COMPARISON, confidence=0.9, year_current=2022, year_prior=2024))
    settings = load_settings(env={"LLM_MODEL": "anthropic/x"})

    bundle, _trace = answer_question(
        "travel comparison", data_dir=tmp_path, documents_dir=tmp_path / "documents", llm_client=client, settings=settings
    )

    assert bundle.status is not AnswerStatus.ERROR
    assert bundle.result is not None
    # The explicit (deliberately non-default) years reached the workflow,
    # not the "two most recent" dataset default (which would be 2024/2023).
    assert bundle.result["years"] == {"current": 2022, "prior": 2024}


def test_model_call_recorded_in_trace(tmp_path, fake_llm_client):
    client = fake_llm_client(
        IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.1),
        provider="anthropic",
        prompt_tokens=42,
        completion_tokens=7,
        latency_ms=99,
        estimated_cost_usd=0.0021,
    )
    settings = load_settings(env={"LLM_MODEL": "anthropic/claude-sonnet-4-5", "LLM_MIN_CONFIDENCE": "0.5"})

    _bundle, trace = answer_question("some question", data_dir=tmp_path, llm_client=client, settings=settings)

    assert trace.model_calls == [
        ModelCall(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-5",
            prompt_tokens=42,
            completion_tokens=7,
            estimated_cost_usd=0.0021,
            latency_ms=99,
        )
    ]
    assert trace.estimated_cost_usd == 0.0021
