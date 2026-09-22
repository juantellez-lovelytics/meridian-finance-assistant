"""Q8 — probable duplicate payment candidates, never a definitive audit.

ANSWER as candidates, bucketed by confidence tier and reversal status, plus
an unconditional statement of what evidence this ledger cannot supply to
confirm an actual double payment (treasury settlement references; reversals
worded differently from the original doc_ref). That statement is present
even with zero candidates — it names a structural limitation of the
dataset, not a property of any particular finding.
"""

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import (
    AnswerStatus,
    Coverage,
    EvidenceBundle,
    MissingEvidence,
    MissingEvidenceReasonCode,
)
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.duplicates import DuplicateDetectionRules, detect_duplicate_candidates
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.tools.vendors import detect_alias_clusters
from finance_assistant.workflows._shared import ToolTrace


def duplicate_payment_check(
    gl: pd.DataFrame,
    vendors: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
    rules: DuplicateDetectionRules | None = None,
) -> EvidenceBundle:
    tt = ToolTrace()
    assumptions: list[str] = []
    if date_start is not None and date_end is not None:
        rows = tt.call(query_ledger, gl, date_start, date_end, date_field=date_field).rows
        assumptions.append(f"scoped to {date_start}..{date_end} on {date_field}")
    else:
        rows = gl
        assumptions.append("no period specified: matched across the full supplied ledger")

    alias_clusters = tt.call(detect_alias_clusters, vendors)
    detection = tt.call(detect_duplicate_candidates, rows, rules=rules, alias_clusters=alias_clusters)

    candidates_by_confidence: dict[str, list[dict]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    reversed_count = 0
    for c in detection.candidates:
        candidates_by_confidence[c.confidence].append(
            {
                "txn_id_a": c.txn_id_a,
                "txn_id_b": c.txn_id_b,
                "amount": c.amount,
                "currency": c.currency,
                "is_reversed": c.is_reversed,
            }
        )
        if c.is_reversed:
            reversed_count += 1

    result = {
        "date_basis": date_field,
        "candidates_by_confidence": candidates_by_confidence,
        "reversed_candidate_count": reversed_count,
        "unreversed_candidate_count": len(detection.candidates) - reversed_count,
        "rules_applied": {"window_days": detection.rules.window_days, "amount_tolerance": detection.rules.amount_tolerance},
    }

    missing_evidence = [
        MissingEvidence(
            what="payment settlement / bank statement reference matching each candidate pair",
            reason=(
                "gl_transactions.csv has no payment-file, bank-statement, or AP settlement-status field; "
                "the fingerprint match is necessary but not sufficient to prove treasury actually settled "
                "the same obligation twice"
            ),
            reason_code=MissingEvidenceReasonCode.OTHER,
        ),
        MissingEvidence(
            what="confirmation that neither leg of a candidate pair was reversed outside what memo-substring matching can detect",
            reason=(
                "is_reversed/reversal_a/reversal_b only find a reversal whose memo explicitly quotes the "
                "original doc_ref string; a differently-worded reversal is not detected, so an unflagged "
                "pair is not proof of a live double payment"
            ),
            reason_code=MissingEvidenceReasonCode.OTHER,
        ),
    ]

    warnings = [detection.limitation, alias_clusters.limitation]
    if reversed_count:
        warnings.append(f"{reversed_count} candidate(s) are already reversed and excluded from any 'live' double-payment risk framing")

    coverage = Coverage.fully_computable(len(rows))

    gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, row_coverage=coverage)

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.DUPLICATE_PAYMENT_CHECK,
        result=result,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        missing_evidence=missing_evidence,
        coverage=coverage,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
