"""Tests for finance_assistant.orchestration.interpreter.

`interpret_with_keywords` is calibrated and tested against hand-written
PARAPHRASES, never against the literal strings in evals/questions.yaml --
tuning a language matcher against the exact phrasing of this project's own
eval set would be the same mistake as hardcoding a dataset value, just
moved into language. A real user/grader asks in their own words. The
literal YAML strings are included too, but only as one additional case per
intent, read from the YAML directly (never duplicated by hand here) --
never the basis the rules are tuned against.
"""

from pathlib import Path

import pytest
import yaml

from finance_assistant.orchestration.intents import Intent, IntentRequest
from finance_assistant.orchestration.interpreter import interpret_with_keywords, interpret_with_llm

QUESTIONS_YAML = yaml.safe_load((Path(__file__).parent.parent / "evals" / "questions.yaml").read_text(encoding="utf-8"))
_LITERAL_CASES = [(case["question"], Intent[case["expected_intent"]]) for case in QUESTIONS_YAML["cases"]]

# The real calibration set: 2-3 hand-written paraphrases per intent, with
# different wording/synonyms/word order than evals/questions.yaml, some
# with the period stated and some without.
_PARAPHRASES = [
    ("what did we spend on operating expenses per cost centre last quarter", Intent.OPEX_BY_COST_CENTRE),
    ("break down opex by cost center for Q3 2024", Intent.OPEX_BY_COST_CENTRE),
    ("how much did we spend on travel last year compared to the year before", Intent.TRAVEL_COMPARISON),
    ("year-over-year T&E spend trend", Intent.TRAVEL_COMPARISON),
    ("what's our total consolidated spend across all entities", Intent.CONSOLIDATED_SPEND),
    ("give me the consolidated USD spend figure for the third quarter", Intent.CONSOLIDATED_SPEND),
    ("show our top vendors sorted by spend", Intent.TOP_VENDORS),
    ("rank our biggest vendors by total spend", Intent.TOP_VENDORS),
    ("are we over or under budget by cost centre this quarter", Intent.BUDGET_VARIANCE),
    ("what's causing the variance vs budget", Intent.BUDGET_VARIANCE),
    ("did any expenses violate our T&E policy", Intent.TE_POLICY_CHECK),
    ("flag any travel expense policy breaches", Intent.TE_POLICY_CHECK),
    ("what's our cost per employee", Intent.HEADCOUNT_COST_PER_FTE),
    ("how much do we spend per FTE", Intent.HEADCOUNT_COST_PER_FTE),
    ("did we accidentally pay any invoice twice", Intent.DUPLICATE_PAYMENT_CHECK),
    ("check for duplicate vendor payments", Intent.DUPLICATE_PAYMENT_CHECK),
]


@pytest.mark.parametrize("question,expected_intent", _PARAPHRASES)
def test_interpret_with_keywords_routes_paraphrases_correctly(question, expected_intent):
    result = interpret_with_keywords(question)
    assert result.intent == expected_intent


@pytest.mark.parametrize("question,expected_intent", _LITERAL_CASES)
def test_interpret_with_keywords_also_handles_the_literal_eval_questions(question, expected_intent):
    """One additional smoke case per intent -- never the calibration basis."""
    result = interpret_with_keywords(question)
    assert result.intent == expected_intent


@pytest.mark.parametrize(
    "question",
    [
        "what's the weather like today",
        "can you recommend a good restaurant nearby",
        "tell me a joke",
    ],
)
def test_interpret_with_keywords_returns_unknown_for_unrecognized_questions(question):
    """Refusing to guess is correct behavior, not a gap."""
    result = interpret_with_keywords(question)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


def test_interpret_with_keywords_extracts_literal_quarter_and_year():
    result = interpret_with_keywords("What was our opex by cost centre in Q2 2023?")
    assert result.quarter == "Q2"
    assert result.year == 2023


def test_interpret_with_keywords_leaves_quarter_and_year_unset_when_absent():
    result = interpret_with_keywords("Who are our top 10 vendors by spend?")
    assert result.quarter is None
    assert result.year is None


def test_interpret_with_llm_builds_model_call_from_fake_completion(fake_llm_client):
    expected = IntentRequest(intent=Intent.CONSOLIDATED_SPEND, confidence=0.87, quarter="Q3")
    client = fake_llm_client(
        expected,
        provider="anthropic",
        prompt_tokens=123,
        completion_tokens=45,
        latency_ms=678,
        estimated_cost_usd=0.0042,
    )

    request, model_call = interpret_with_llm("What was our consolidated spend in Q3?", model="anthropic/claude-sonnet-4-5", client=client)

    assert request == expected
    assert model_call.provider == "anthropic"
    assert model_call.model == "anthropic/claude-sonnet-4-5"
    assert model_call.prompt_tokens == 123
    assert model_call.completion_tokens == 45
    assert model_call.latency_ms == 678
    assert model_call.estimated_cost_usd == 0.0042


def test_interpret_with_llm_passes_through_unknown_cost(fake_llm_client):
    expected = IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.6)
    client = fake_llm_client(expected, estimated_cost_usd="unknown")

    _request, model_call = interpret_with_llm("some question", model="openai/gpt-4o-mini", client=client)

    assert model_call.estimated_cost_usd == "unknown"
