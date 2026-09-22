"""The Evidence Gate — a deterministic function, never a prompt.

Applies the master prompt's degradation table exactly:

  A. missing FX rate affects the requested total  -> PARTIAL, or REFUSED if
     the workflow declares force_refuse_incomplete_total (the total asked
     for has no honest partial answer — e.g. Q3's exact consolidated total)
  B. missing the denominator of a ratio            -> REFUSED
  C. row coverage < 100% on a consolidated total    -> ceiling of PARTIAL
  D. ambiguous period, readings materially differ   -> NEEDS_CLARIFICATION
  E. non-authoritative grouping would change result -> ceiling of PARTIAL
  F. no data for the requested period               -> REFUSED

Input is explicit typed facts, not inspection of a draft EvidenceBundle:
inspecting a bundle would force this module to infer "is a denominator
missing?" from loosely-typed result/warnings content, which is both fragile
and impossible to unit test one table row at a time. "El modelo no
participa en esta decisión" applies here in the strongest sense — this
function has no branch conditioned on Intent, LLM output, or any dataset
value; it only ever sees the facts a workflow already computed.

Severity: REFUSED > NEEDS_CLARIFICATION > PARTIAL > ANSWER. REFUSED always
wins: if the data is already unusable (no denominator, no data for the
period), asking which reading was meant is moot — every reading is equally
unusable. Rows C and E are ceilings — they can only ever pull an ANSWER
draft down to PARTIAL, never escalate further; rows A/B/D/F assert a status
outright. When several REFUSED-triggering rows fire at once, every reason is
kept (concatenated), never trimmed to one — a transparent refusal should be
maximally transparent, not truncated.
"""

from dataclasses import dataclass

from finance_assistant.evidence.models import AnswerStatus, Coverage

# Same relative-difference-with-zero-guard shape as
# tools/budget.py::PLAUSIBILITY_AMBIGUITY_TOLERANCE — a conceptually
# independent business rule (period-reading materiality, not budget
# duplicate-key plausibility) that happens to share a sensible default.
DEFAULT_PERIOD_READING_TOLERANCE = 0.05

_SEVERITY = {
    AnswerStatus.ANSWER: 0,
    AnswerStatus.PARTIAL: 1,
    AnswerStatus.NEEDS_CLARIFICATION: 2,
    AnswerStatus.REFUSED: 3,
}


@dataclass(frozen=True)
class GateResult:
    final_status: AnswerStatus
    warnings_added: list[str]
    refusal_reason: str | None
    clarification_options: list[str]
    firing_conditions: list[str]


def readings_are_material_difference(
    readings: list[tuple[str, AnswerStatus | None, float | None]],
    tolerance: float = DEFAULT_PERIOD_READING_TOLERANCE,
) -> bool:
    if len(readings) < 2:
        return False

    statuses = {status for _, status, _ in readings if status is not None}
    if len(statuses) > 1:
        return True

    values = [magnitude for _, _, magnitude in readings if magnitude is not None]
    if len(values) < 2:
        return False

    spread = max(values) - min(values)
    reference = max(abs(v) for v in values)
    if reference == 0:
        return spread != 0
    return (spread / reference) > tolerance


def _max_status(a: AnswerStatus, b: AnswerStatus) -> AnswerStatus:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


def apply_gate(
    draft_status: AnswerStatus,
    *,
    fx_coverage: Coverage | None = None,
    missing_fx_affects_requested_total: bool = False,
    missing_denominator: bool = False,
    row_coverage: Coverage | None = None,
    period_readings: list[tuple[str, AnswerStatus | None, float | None]] | None = None,
    period_reading_tolerance: float = DEFAULT_PERIOD_READING_TOLERANCE,
    grouping_would_change_result: bool = False,
    no_data_for_period: bool = False,
    force_refuse_incomplete_total: bool = False,
) -> GateResult:
    warnings_added: list[str] = []
    refusal_reasons: list[str] = []
    clarification_options: list[str] = []
    firing_conditions: list[str] = []

    status = draft_status
    ceiling_fired = False

    # Row A — missing FX rate affects the requested total.
    if missing_fx_affects_requested_total:
        firing_conditions.append("missing_fx_affects_requested_total")
        if force_refuse_incomplete_total:
            status = _max_status(status, AnswerStatus.REFUSED)
            refusal_reasons.append("a required FX rate is missing and affects the requested total")
            warnings_added.append("a required FX rate is missing and affects the requested total (forced REFUSED)")
        else:
            status = _max_status(status, AnswerStatus.PARTIAL)
            warnings_added.append("a required FX rate is missing and affects the requested total")

    # Row B — missing denominator.
    if missing_denominator:
        firing_conditions.append("missing_denominator")
        status = _max_status(status, AnswerStatus.REFUSED)
        refusal_reasons.append("the denominator required for this ratio is not available in the dataset")
        warnings_added.append("the denominator required for this ratio is not available in the dataset")

    # Row C — row coverage < 100% on a consolidated total (ceiling only).
    # fx_coverage and row_coverage are independent completeness dimensions
    # (FX-convertibility vs, e.g., account-hierarchy mapping) — either one
    # alone is enough to cap an ANSWER draft at PARTIAL. This is what makes
    # "coverage < 100% => never ANSWER" unconditional: it fires even when a
    # workflow didn't separately flag missing_fx_affects_requested_total.
    for label, coverage in (("fx", fx_coverage), ("row", row_coverage)):
        if coverage is not None and coverage.computable_amount_pct < 100.0:
            firing_conditions.append(f"{label}_coverage_incomplete")
            ceiling_fired = True
            warnings_added.append(
                f"{label} coverage is incomplete ({coverage.computable_rows}/{coverage.selected_rows} rows computable)"
            )

    # Row D — ambiguous period, materially different readings.
    if period_readings and readings_are_material_difference(period_readings, tolerance=period_reading_tolerance):
        firing_conditions.append("ambiguous_period")
        status = _max_status(status, AnswerStatus.NEEDS_CLARIFICATION)
        clarification_options = [label for label, _, _ in period_readings]
        warnings_added.append("candidate period readings differ materially; clarification is required")

    # Row E — non-authoritative grouping decision would change the result (ceiling only).
    if grouping_would_change_result:
        firing_conditions.append("grouping_would_change_result")
        ceiling_fired = True
        warnings_added.append("a non-authoritative grouping decision would change the result")

    # Row F — no data for the requested period.
    if no_data_for_period:
        firing_conditions.append("no_data_for_period")
        status = _max_status(status, AnswerStatus.REFUSED)
        refusal_reasons.append("no data exists for the requested period")
        warnings_added.append("no data exists for the requested period")

    if ceiling_fired and status == AnswerStatus.ANSWER:
        status = AnswerStatus.PARTIAL

    if status != AnswerStatus.REFUSED:
        refusal_reasons = []
    if status != AnswerStatus.NEEDS_CLARIFICATION:
        clarification_options = []

    return GateResult(
        final_status=status,
        warnings_added=warnings_added,
        refusal_reason="; ".join(refusal_reasons) if refusal_reasons else None,
        clarification_options=clarification_options,
        firing_conditions=firing_conditions,
    )
