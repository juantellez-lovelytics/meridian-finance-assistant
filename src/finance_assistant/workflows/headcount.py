"""Q7 — headcount cost per FTE. Always REFUSED — the other clean refusal.

The refusal is structural, not data-dependent: FTE is a denominator that
does not exist in any of the five datasets, confirmed both mechanically
(scanning every declared schema for an fte/headcount column) and by a real
search_documents retrieval of the board memo's "Headcount" section — not a
hardcoded citation, which is the entire point of having tools/documents.py.
"""

import re
from pathlib import Path

from finance_assistant import config
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import (
    AnswerStatus,
    Coverage,
    EvidenceBundle,
    MissingEvidence,
    MissingEvidenceReasonCode,
    SourceRef,
)
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.documents import search_documents
from finance_assistant.workflows._shared import ToolTrace

_FTE_PATTERN = re.compile(r"fte|headcount", re.IGNORECASE)
_SCHEMAS = [config.GL_SCHEMA, config.COA_SCHEMA, config.BUDGET_SCHEMA, config.FX_SCHEMA, config.VENDORS_SCHEMA]


def _scan_schemas_for_fte_column() -> list[str]:
    hits: list[str] = []
    for schema in _SCHEMAS:
        for column in schema["required_columns"]:
            if _FTE_PATTERN.search(column):
                hits.append(f"{schema['filename']}::{column}")
    return hits


def headcount_cost_per_fte(documents_dir: str | Path | None = None) -> EvidenceBundle:
    tt = ToolTrace()
    fte_columns = _scan_schemas_for_fte_column()

    search_result = tt.call(
        search_documents,
        query="headcount FTE", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir, max_results=1
    )

    sources: list[SourceRef] = []
    warnings: list[str] = []
    citation = None
    if search_result.matches:
        citation = SourceRef.from_document_match(search_result.matches[0])
        sources.append(citation)
    else:
        warnings.append("board_memo_2024_q2.md returned no section discussing headcount/FTE")

    schema_names = ", ".join(s["filename"] for s in _SCHEMAS)
    missing_evidence = [
        MissingEvidence(
            what="FTE headcount per cost centre or entity (the denominator for cost-per-FTE)",
            reason=(
                f"not a column in any of the five declared dataset schemas ({schema_names}); "
                + (f"confirmed no fte/headcount-named column exists (found: {fte_columns})" if not fte_columns else f"unexpected match(es): {fte_columns}")
            ),
            reason_code=MissingEvidenceReasonCode.MISSING_FTE_DENOMINATOR,
            citation=citation,
        )
    ]

    if citation:
        refusal_reason = (
            f"the FTE denominator does not exist in the finance ledger; {citation.filename} section "
            f"{citation.section!r} confirms headcount is tracked separately by the People team in the HR system"
        )
    else:
        refusal_reason = "the FTE denominator does not exist in the finance ledger, and no dataset or document in scope defines one"

    gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, missing_denominator=True)

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.HEADCOUNT_COST_PER_FTE,
        result=None,
        sources=sources,
        warnings=warnings + gate_result.warnings_added,
        missing_evidence=missing_evidence,
        coverage=Coverage(selected_rows=0, computable_rows=0, computable_amount_pct=0.0),
        refusal_reason=refusal_reason,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
