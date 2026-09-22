"""The orchestrator: free text -> (EvidenceBundle, RunTrace).

Wires together the Question Interpreter, the Plan Registry, a workflow,
and the trace builder. The model never decides `status` (that stays
`evidence.gate.apply_gate`, called inside each workflow), never sees a
DataFrame, and never computes a number -- this module's own job is purely
routing: interpret -> enforce ceilings -> resolve params -> run -> trace.

Does not write the trace to disk itself (`write_trace` stays a caller
concern, same as the JSON CLI path) -- keeps this function pure and easy
to test without touching the filesystem.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from finance_assistant import config
from finance_assistant.evidence.models import AnswerStatus, Coverage, EvidenceBundle
from finance_assistant.evidence.trace import ModelCall, RunTrace, build_trace
from finance_assistant.orchestration.interpreter import (
    INTENT_DESCRIPTIONS,
    NEARBY_INTENTS,
    LiteLLMClient,
    LLMClient,
    interpret_with_keywords,
    interpret_with_llm,
)
from finance_assistant.orchestration.intents import Intent, IntentRequest
from finance_assistant.orchestration.plans import (
    REGISTRY,
    PlanResolutionError,
    assemble_kwargs,
    default_params,
    extract_llm_params,
    load_dataframes,
)
from finance_assistant.orchestration.settings import Settings, load_settings


def _error_bundle(intent: Intent, assumptions: list[str], reason: str) -> EvidenceBundle:
    # ERROR's reason goes in `warnings`, never `refusal_reason` -- the
    # EvidenceBundle validator forbids refusal_reason unless status==REFUSED.
    return EvidenceBundle(
        status=AnswerStatus.ERROR,
        intent=intent,
        result=None,
        coverage=Coverage(selected_rows=0, computable_rows=0, computable_amount_pct=0.0),
        assumptions=assumptions,
        warnings=[reason],
    )


def _clarification_options(intent: Intent) -> list[str]:
    if intent is Intent.UNKNOWN:
        return [INTENT_DESCRIPTIONS[i] for i in Intent if i is not Intent.UNKNOWN]
    return [INTENT_DESCRIPTIONS[intent], *(INTENT_DESCRIPTIONS[i] for i in NEARBY_INTENTS[intent])]


def _needs_clarification_bundle(intent: Intent, assumptions: list[str]) -> EvidenceBundle:
    return EvidenceBundle(
        status=AnswerStatus.NEEDS_CLARIFICATION,
        intent=intent,
        result=None,
        coverage=Coverage.fully_computable(0),
        assumptions=assumptions,
        clarification_options=_clarification_options(intent),
    )


def _check_ceilings(
    model_calls: list[ModelCall], settings: Settings, *, intent: Intent, assumptions: list[str]
) -> EvidenceBundle | None:
    if len(model_calls) > settings.max_calls_per_question:
        return _error_bundle(
            intent, assumptions, f"model call ceiling exceeded: {len(model_calls)} call(s) > max {settings.max_calls_per_question}"
        )

    total_tokens = sum(c.prompt_tokens + c.completion_tokens for c in model_calls)
    if total_tokens > settings.max_tokens_per_question:
        return _error_bundle(
            intent,
            assumptions,
            f"model token ceiling exceeded: {total_tokens} token(s) > max {settings.max_tokens_per_question}",
        )

    known_costs = [c.estimated_cost_usd for c in model_calls if c.estimated_cost_usd != "unknown"]
    total_known_cost = sum(known_costs) if known_costs else 0.0
    if total_known_cost > settings.max_cost_usd_per_question:
        return _error_bundle(
            intent,
            assumptions,
            f"model cost ceiling exceeded: ${total_known_cost:.4f} > max ${settings.max_cost_usd_per_question:.2f}",
        )

    if any(c.estimated_cost_usd == "unknown" for c in model_calls):
        assumptions.append(f"model pricing unknown for {settings.llm_model} — cost ceiling could not be fully enforced")

    return None


def answer_question(
    question: str,
    *,
    data_dir: Path | None = None,
    documents_dir: Path | None = None,
    model: str | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[EvidenceBundle, RunTrace]:
    settings = settings or load_settings(model=model)
    data_dir = data_dir or config.DATA_DIR
    documents_dir = documents_dir or data_dir / "documents"

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    assumptions: list[str] = []
    model_calls: list[ModelCall] = []

    def _finish(bundle: EvidenceBundle, date_basis: str = config.DEFAULT_FINANCIAL_DATE_FIELD) -> tuple[EvidenceBundle, RunTrace]:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        trace = build_trace(
            question=question, bundle=bundle, started_at=started_at, duration_ms=duration_ms, date_basis=date_basis, model_calls=model_calls
        )
        return bundle, trace

    # --- Interpret -----------------------------------------------------
    request: IntentRequest
    if llm_client is not None:
        try:
            request, model_call = interpret_with_llm(question, model=settings.llm_model, client=llm_client)
            model_calls.append(model_call)
        except Exception as exc:
            return _finish(_error_bundle(Intent.UNKNOWN, assumptions, f"LLM call failed: {exc}"))
    elif settings.has_credential():
        try:
            request, model_call = interpret_with_llm(question, model=settings.llm_model, client=LiteLLMClient())
            model_calls.append(model_call)
        except Exception as exc:
            return _finish(_error_bundle(Intent.UNKNOWN, assumptions, f"LLM call failed: {exc}"))
    else:
        request = interpret_with_keywords(question)
        assumptions.append(f"intent interpreted via keyword fallback (no LLM credential detected for {settings.llm_model})")

    # --- Ceilings --------------------------------------------------------
    ceiling_error = _check_ceilings(model_calls, settings, intent=request.intent, assumptions=assumptions)
    if ceiling_error is not None:
        return _finish(ceiling_error)

    # --- Confidence / unknown gate ---------------------------------------
    if request.intent is Intent.UNKNOWN or request.confidence < settings.min_confidence:
        return _finish(_needs_clarification_bundle(request.intent, assumptions))

    # --- Resolve plan and run the workflow --------------------------------
    spec = REGISTRY[request.intent]
    try:
        dataframes = load_dataframes(spec, data_dir)
        params = {**default_params(request.intent, dataframes), **extract_llm_params(request, spec)}
        kwargs = assemble_kwargs(spec, params, dataframes, documents_dir)
    except PlanResolutionError as exc:
        return _finish(_error_bundle(request.intent, assumptions, str(exc)))

    bundle = spec.workflow(**kwargs)
    if assumptions:
        bundle = bundle.model_copy(update={"assumptions": [*assumptions, *bundle.assumptions]})

    date_basis = kwargs.get("date_field", config.DEFAULT_FINANCIAL_DATE_FIELD)
    return _finish(bundle, date_basis=date_basis)
