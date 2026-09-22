"""Q6 — T&E policy candidates, never a definitive audit.

ANSWER as candidates, bucketed by rule and by state. `evaluate_te_policy`'s
five-state model already carries every row-level uncertainty honestly
(INSUFFICIENT_EVIDENCE is itself the transparent answer for what can't be
confirmed) — degrading the aggregate status on top of that would conflate
"some individual findings are inconclusive" (expected, already reported)
with "the aggregate answer is incomplete" (it isn't: every perimeter row
got evaluated).
"""

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import AnswerStatus, Coverage, EvidenceBundle, SourceRef
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.documents import search_documents
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.tools.te_policy import evaluate_te_policy, load_policy_rules
from finance_assistant.workflows._shared import ToolTrace


def te_policy_check(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    date_start: str,
    date_end: str,
    perimeter_basis: str = "reported",
    policy_path: str | None = None,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> EvidenceBundle:
    tt = ToolTrace()
    ledger = tt.call(query_ledger, gl, date_start, date_end, date_field=date_field)
    policy = tt.call(load_policy_rules, policy_path)
    evaluation = tt.call(evaluate_te_policy, ledger.rows, coa, fx, policy, perimeter_basis=perimeter_basis, date_field=date_field)

    findings_by_rule: dict[str, dict[str, int]] = {}
    findings_by_state: dict[str, int] = {}
    for finding in evaluation.findings:
        findings_by_rule.setdefault(finding.rule_id, {}).setdefault(finding.state.value, 0)
        findings_by_rule[finding.rule_id][finding.state.value] += 1
        findings_by_state[finding.state.value] = findings_by_state.get(finding.state.value, 0) + 1

    # Backfill real citations: PolicyFinding only carries source_document/
    # source_section as bare strings, not a real DocumentMatch.
    cited_sections = sorted({(f.source_document, f.source_section) for f in evaluation.findings if f.state.value != "NOT_APPLICABLE"})
    sources: list[SourceRef] = []
    for document, section in cited_sections:
        search_result = tt.call(search_documents, query=section, filenames=[document], max_results=1)
        if search_result.matches:
            sources.append(SourceRef.from_document_match(search_result.matches[0]))

    assumptions = [
        f"date basis: {date_field}",
        f"T&E perimeter basis: {perimeter_basis!r} "
        f"({evaluation.rows_excluded_by_reclassification.row_count} row(s) excluded under this basis "
        "that would be included under the alternative)",
    ]

    warnings = list(evaluation.warnings)
    if evaluation.unmapped_account_rows:
        warnings.append(f"{evaluation.unmapped_account_rows} row(s) had an unmapped account_code and could not be evaluated")
    warnings.append(evaluation.limitation)

    result = {
        "date_basis": date_field,
        "perimeter_basis": evaluation.perimeter_basis,
        "perimeter_rows": evaluation.perimeter_rows,
        "total_rows": evaluation.total_rows,
        "findings_by_rule": findings_by_rule,
        "findings_by_state": findings_by_state,
    }

    coverage = Coverage.fully_computable(evaluation.perimeter_rows)

    gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, row_coverage=coverage)

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.TE_POLICY_CHECK,
        result=result,
        sources=sources,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        coverage=coverage,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
