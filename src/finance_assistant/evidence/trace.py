"""RunTrace — one JSON per execution in `traces/`, per docs/PROMPT_MAESTRO.md's
TRACE section.

Built entirely from an already-finished `EvidenceBundle`: `steps[]` is a
readable projection of `bundle.tool_calls` (populated by
`workflows._shared.ToolTrace`) followed by `bundle.calculations`, never a
second, independent instrumentation path. `final_evidence` is the bundle
itself, untouched — the truncation in `evidence.summarize.summarize_for_trace`
only applies when building `steps[]`, since a step must "fit on screen"
while the bundle already keeps its own contents at aggregate, non-row-level
granularity.

`model_calls` is populated by `orchestration.orchestrator.answer_question`
when a question went through the LLM interpreter (or stays empty when the
keyword fallback answered it instead); JSON-mode CLI callers never pass it,
so their trace keeps the original `model_calls=[]`, `estimated_cost_usd=0.0`
shape unchanged. Per-step `duration_ms` for a "tool" step is the real
wall-clock time `ToolTrace` measured around that call; for a "calc" step it
stays 0 -- `CalcStep` arithmetic is inline Python with no timer, matching
the master prompt's own example.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from finance_assistant.evidence.models import AnswerStatus, EvidenceBundle
from finance_assistant.orchestration.intents import Intent
from finance_assistant.evidence.summarize import summarize_for_trace


class Step(BaseModel):
    step: int
    type: Literal["tool", "calc"]
    name: str
    arguments: dict[str, object]
    result_summary: dict[str, object]
    duration_ms: int = 0


class ModelCall(BaseModel):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | Literal["unknown"]
    latency_ms: int


class RunTrace(BaseModel):
    run_id: str
    started_at: str
    question: str
    status: AnswerStatus
    date_basis: str
    steps: list[Step] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    final_evidence: dict
    duration_ms: int
    estimated_cost_usd: float | Literal["unknown"]


def _build_steps(bundle: EvidenceBundle) -> list[Step]:
    steps: list[Step] = []
    n = 0

    for tc in bundle.tool_calls:
        n += 1
        steps.append(
            Step(
                step=n,
                type="tool",
                name=tc.tool,
                arguments=tc.arguments_summary,
                result_summary=tc.result_summary,
                duration_ms=tc.duration_ms,
            )
        )

    for calc in bundle.calculations:
        n += 1
        arguments = summarize_for_trace({"operation": calc.operation, **calc.inputs})
        result_summary = summarize_for_trace({"output": calc.output})
        steps.append(
            Step(
                step=n,
                type="calc",
                name=calc.description,
                arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                result_summary=result_summary if isinstance(result_summary, dict) else {"value": result_summary},
            )
        )

    return steps


def _make_run_id(started_at: datetime, intent: Intent) -> str:
    return f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_{intent.value}_{uuid4().hex[:8]}"


def _total_estimated_cost(model_calls: list[ModelCall]) -> float | Literal["unknown"]:
    if not model_calls:
        return 0.0
    if any(call.estimated_cost_usd == "unknown" for call in model_calls):
        return "unknown"
    return sum(call.estimated_cost_usd for call in model_calls)  # type: ignore[misc]


def build_trace(
    *,
    question: str,
    bundle: EvidenceBundle,
    started_at: datetime,
    duration_ms: int,
    date_basis: str,
    model_calls: list[ModelCall] | None = None,
) -> RunTrace:
    calls = model_calls or []
    return RunTrace(
        run_id=_make_run_id(started_at, bundle.intent),
        started_at=started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        question=question,
        status=bundle.status,
        date_basis=date_basis,
        steps=_build_steps(bundle),
        model_calls=calls,
        final_evidence=bundle.model_dump(mode="json"),
        duration_ms=duration_ms,
        estimated_cost_usd=_total_estimated_cost(calls),
    )


def write_trace(trace: RunTrace, traces_dir: Path) -> Path:
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace.run_id}.json"
    path.write_text(json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
