"""The Question Interpreter: converts free-text into an `IntentRequest`.

Two implementations share one output type:

- `interpret_with_llm` -- exactly one LLM call, structured Pydantic output.
  The model sees ONLY the question text and the intent catalogue below --
  never ledger rows, never cost-centre/vendor lists, never which years or
  periods are actually present in any dataset. It classifies and extracts
  only what the text explicitly states; it never computes, ranks, or
  decides a status.
- `interpret_with_keywords` -- a deterministic fallback used only when no
  LLM credential is available (see orchestration.settings.Settings). It is
  a continuity mechanism, not a substitute interpreter -- see its own
  docstring for the scope this implies.
"""

import re
import time
from dataclasses import dataclass
from typing import Protocol

from finance_assistant.evidence.trace import ModelCall
from finance_assistant.orchestration.intents import Intent, IntentRequest

# Phrased as a plain-language question a user might ask, not a technical
# parameter spec. Dual-purpose: feeds the LLM system prompt below AND is
# reused verbatim as the text behind `clarification_options`
# (orchestrator.py) -- one wording, never two copies to drift apart.
INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.OPEX_BY_COST_CENTRE: "What was opex by cost centre for a given quarter?",
    Intent.TRAVEL_COMPARISON: "How does travel & entertainment spend in one year compare to another?",
    Intent.CONSOLIDATED_SPEND: "What was total consolidated spend across entities, in USD, for a quarter?",
    Intent.TOP_VENDORS: "Who are the top vendors by spend over a date range?",
    Intent.BUDGET_VARIANCE: "How did cost centres perform against budget for a quarter, and what drove it?",
    Intent.TE_POLICY_CHECK: "Are there any candidate travel & entertainment policy breaches in a date range?",
    Intent.HEADCOUNT_COST_PER_FTE: "What is headcount cost per FTE?",
    Intent.DUPLICATE_PAYMENT_CHECK: "Are there any candidate duplicate vendor payments?",
    Intent.UNKNOWN: "None of the above, or the question is unclear / out of scope.",
}

# A small, hand-curated "commonly confused with" map used only to build
# `clarification_options` when a real intent was detected but confidence
# is low. UX copy, not a data fact -- deliberately static (only one LLM
# call is allowed per question, so there is no second call to rank
# alternatives).
NEARBY_INTENTS: dict[Intent, tuple[Intent, Intent]] = {
    Intent.OPEX_BY_COST_CENTRE: (Intent.CONSOLIDATED_SPEND, Intent.BUDGET_VARIANCE),
    Intent.TRAVEL_COMPARISON: (Intent.TE_POLICY_CHECK, Intent.OPEX_BY_COST_CENTRE),
    Intent.CONSOLIDATED_SPEND: (Intent.OPEX_BY_COST_CENTRE, Intent.BUDGET_VARIANCE),
    Intent.TOP_VENDORS: (Intent.DUPLICATE_PAYMENT_CHECK, Intent.TE_POLICY_CHECK),
    Intent.BUDGET_VARIANCE: (Intent.OPEX_BY_COST_CENTRE, Intent.CONSOLIDATED_SPEND),
    Intent.TE_POLICY_CHECK: (Intent.TRAVEL_COMPARISON, Intent.DUPLICATE_PAYMENT_CHECK),
    Intent.HEADCOUNT_COST_PER_FTE: (Intent.OPEX_BY_COST_CENTRE, Intent.BUDGET_VARIANCE),
    Intent.DUPLICATE_PAYMENT_CHECK: (Intent.TOP_VENDORS, Intent.TE_POLICY_CHECK),
}

_SYSTEM_PROMPT = (
    "You are the question interpreter for a finance analyst assistant. You are given "
    "ONLY the user's question text -- never ledger data, account lists, vendor lists, "
    "cost-centre lists, or which years/periods are present in any dataset. Classify the "
    "question into exactly one of the intents below and extract only the parameters the "
    "question text EXPLICITLY states (a literal quarter, a literal year, a literal date). "
    "Never guess or infer a year/date the text doesn't state -- leave that field unset "
    "instead; a downstream deterministic step resolves sensible defaults from the real "
    "dataset. If no intent clearly matches, or you are not confident, return "
    'intent="unknown" with a low confidence.\n\n'
    + "\n".join(f"- {intent.value}: {description}" for intent, description in INTENT_DESCRIPTIONS.items())
)


class LLMClient(Protocol):
    def complete(
        self, *, model: str, system: str, user: str, response_model: type[IntentRequest]
    ) -> "LLMCompletion": ...


@dataclass(frozen=True)
class LLMCompletion:
    parsed: IntentRequest
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    estimated_cost_usd: float | str  # "unknown" sentinel, never a made-up number


class LiteLLMClient:
    """The only class importing `litellm` -- imported lazily inside
    `complete()` so importing this module (e.g. to reach
    `interpret_with_keywords`) never requires litellm to succeed against a
    real/absent credential."""

    def complete(
        self, *, model: str, system: str, user: str, response_model: type[IntentRequest]
    ) -> LLMCompletion:
        import litellm

        t0 = time.perf_counter()
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=response_model,
            )
        except litellm.BadRequestError:
            # Some providers' strict tool-schema enforcement (observed on
            # Groq's llama tool-calling) rejects the call outright when the
            # model omits a null-valued optional property from its tool
            # arguments, even though the schema marks it nullable. Fall back
            # to plain JSON-object mode with the schema spelled out in the
            # prompt instead of enforced server-side; `response_model` already
            # tolerates missing optional fields on parse, so this degrades
            # gracefully rather than surfacing as an interpreter failure.
            schema_prompt = (
                f"{system}\n\nRespond with ONLY a JSON object matching this JSON schema. "
                f"Omit any field you are not confident about.\n{response_model.model_json_schema()}"
            )
            response = litellm.completion(
                model=model,
                messages=[{"role": "system", "content": schema_prompt}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed = response_model.model_validate_json(response.choices[0].message.content)
        try:
            # `model=` must be passed explicitly: `response.model` echoes back
            # the provider's raw model name without the "groq/"-style prefix
            # litellm's cost lookup needs, which otherwise raises even when
            # pricing for this exact model is in litellm.model_cost.
            cost = litellm.completion_cost(completion_response=response, model=model)
        except Exception:
            cost = "unknown"
        return LLMCompletion(
            parsed=parsed,
            provider=model.split("/", 1)[0],
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )


def interpret_with_llm(question: str, *, model: str, client: LLMClient) -> tuple[IntentRequest, ModelCall]:
    completion = client.complete(model=model, system=_SYSTEM_PROMPT, user=question, response_model=IntentRequest)
    model_call = ModelCall(
        provider=completion.provider,
        model=completion.model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        estimated_cost_usd=completion.estimated_cost_usd,
        latency_ms=completion.latency_ms,
    )
    return completion.parsed, model_call


_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Ordered, first-match-wins. More specific patterns first, so e.g. a
# question about T&E "policy"/"breach" resolves to TE_POLICY_CHECK rather
# than TRAVEL_COMPARISON (both can mention "travel"), and "against budget"
# resolves to BUDGET_VARIANCE before OPEX's "cost centre" pattern gets a
# chance (a budget-variance question also names cost centres).
_KEYWORD_RULES: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.DUPLICATE_PAYMENT_CHECK, ("duplicate", "paid twice", "twice", "double pay")),
    (Intent.TE_POLICY_CHECK, ("t&e policy", "policy", "breach")),
    (Intent.HEADCOUNT_COST_PER_FTE, ("per fte", "fte", "headcount", "per employee", "per head")),
    (Intent.BUDGET_VARIANCE, ("against budget", "vs budget", "budget variance", "budget")),
    (Intent.TOP_VENDORS, ("top vendor", "top 10 vendor", "vendors by spend", "biggest vendor")),
    (Intent.TRAVEL_COMPARISON, ("travel & entertainment spend", "travel spend", "travel", "t&e spend")),
    (Intent.CONSOLIDATED_SPEND, ("consolidated",)),
    (Intent.OPEX_BY_COST_CENTRE, ("opex", "cost centre", "cost center")),
]


def interpret_with_keywords(question: str) -> IntentRequest:
    """Deterministic continuity mechanism for when no LLM credential is
    available (see orchestration.settings.Settings.has_credential) -- NOT
    a substitute for the real interpreter. It recognizes a fixed,
    deliberately narrow set of keyword patterns; a question phrased in a
    way it doesn't recognize correctly returns UNKNOWN with confidence
    0.0 rather than guessing. It must never be tuned against the literal
    strings in evals/questions.yaml -- calibrating a language matcher
    against the exact phrasing of this project's own eval set is the same
    mistake as hardcoding a dataset value, just moved into language; a
    real user/grader will ask in their own words. Calibration lives in
    tests/test_orchestration_interpreter.py as hand-written paraphrases."""
    text = question.lower()
    matched = next((intent for intent, patterns in _KEYWORD_RULES if any(p in text for p in patterns)), Intent.UNKNOWN)
    quarter_match = _QUARTER_RE.search(question)
    year_match = _YEAR_RE.search(question)
    return IntentRequest(
        intent=matched,
        confidence=1.0 if matched is not Intent.UNKNOWN else 0.0,
        quarter=f"Q{quarter_match.group(1)}" if quarter_match else None,
        year=int(year_match.group(0)) if year_match else None,
    )
