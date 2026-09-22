"""Evidence model — the structure every workflow returns instead of prose.

Per the master prompt's evidence model: a workflow never returns text, it
returns an `EvidenceBundle`. The renderer (a later, LLM-touching phase) may
only use values already inside that bundle. `AnswerStatus` is set entirely
by `evidence.gate.apply_gate` from typed facts — never by this module, and
never by an LLM ("el modelo no participa en esta decisión"). The
`@model_validator` here checks structural self-consistency of an
already-decided bundle (does a REFUSED bundle carry a reason?), which is a
different job from the gate's: deciding what status the facts justify.

This is the first pydantic usage in the codebase (tools/ stays stdlib
dataclasses by convention; pydantic is reserved for this evidence layer).
"""

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from finance_assistant.orchestration.intents import Intent

if TYPE_CHECKING:
    from finance_assistant.tools.documents import DocumentMatch
    from finance_assistant.tools.fx import FxCoverage


class AnswerStatus(str, Enum):
    ANSWER = "answer"
    PARTIAL = "partial"
    REFUSED = "refused"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"


class Coverage(BaseModel):
    selected_rows: int
    computable_rows: int
    computable_amount_pct: float

    @classmethod
    def from_fx_coverage(cls, fx_coverage: "FxCoverage") -> "Coverage":
        pct = (
            fx_coverage.convertible_rows / fx_coverage.selected_rows * 100.0
            if fx_coverage.selected_rows
            else 100.0
        )
        return cls(
            selected_rows=fx_coverage.selected_rows,
            computable_rows=fx_coverage.convertible_rows,
            computable_amount_pct=pct,
        )

    @classmethod
    def fully_computable(cls, selected_rows: int) -> "Coverage":
        """Convention: 0 rows selected = 100% (vacuously complete) — for when
        a workflow's own filter genuinely leaves nothing to compute (e.g. an
        empty perimeter). When "coverage" instead means "the required data
        source doesn't exist at all" (e.g. headcount_cost_per_fte), build
        Coverage(0, 0, 0.0) directly instead of using this constructor — 0%
        reflects "no computation was possible" better than "vacuously
        complete"."""
        return cls(selected_rows=selected_rows, computable_rows=selected_rows, computable_amount_pct=100.0)

    @classmethod
    def combine(cls, *coverages: "Coverage") -> "Coverage":
        selected = sum(c.selected_rows for c in coverages)
        computable = sum(c.computable_rows for c in coverages)
        pct = (computable / selected * 100.0) if selected else 100.0
        return cls(selected_rows=selected, computable_rows=computable, computable_amount_pct=pct)


class SourceRef(BaseModel):
    filename: str
    section: str
    snippet: str
    evidence_id: str

    @classmethod
    def from_document_match(cls, match: "DocumentMatch") -> "SourceRef":
        return cls(filename=match.filename, section=match.section, snippet=match.snippet, evidence_id=match.evidence_id)


class MissingEvidenceReasonCode(str, Enum):
    MISSING_FX_RATE = "missing_fx_rate"
    MISSING_FTE_DENOMINATOR = "missing_fte_denominator"
    NO_DATA_FOR_PERIOD = "no_data_for_period"
    MISSING_APPROVAL_EVIDENCE = "missing_approval_evidence"
    OTHER = "other"


class MissingEvidence(BaseModel):
    what: str
    reason: str
    reason_code: MissingEvidenceReasonCode
    citation: SourceRef | None = None


class CalcStep(BaseModel):
    description: str
    operation: str  # e.g. "sum", "groupby_sum", "variance_pct"
    inputs: dict[str, object]
    output: object


class ToolCall(BaseModel):
    tool: str
    arguments_summary: dict[str, object]
    result_summary: dict[str, object]
    duration_ms: int = 0


class EvidenceBundle(BaseModel):
    status: AnswerStatus
    intent: Intent
    result: dict | None
    sources: list[SourceRef] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list)
    coverage: Coverage
    calculations: list[CalcStep] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    refusal_reason: str | None = None
    clarification_options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_consistency(self) -> "EvidenceBundle":
        if self.status == AnswerStatus.REFUSED and not self.refusal_reason:
            raise ValueError("status=REFUSED requires a non-empty refusal_reason")
        if self.status != AnswerStatus.REFUSED and self.refusal_reason:
            raise ValueError("refusal_reason is only allowed when status=REFUSED")
        if self.status == AnswerStatus.NEEDS_CLARIFICATION and not self.clarification_options:
            raise ValueError("status=NEEDS_CLARIFICATION requires at least one clarification_options entry")
        if self.status in (AnswerStatus.NEEDS_CLARIFICATION, AnswerStatus.REFUSED) and self.result is not None:
            raise ValueError(f"status={self.status.value} must not carry a result")
        return self
