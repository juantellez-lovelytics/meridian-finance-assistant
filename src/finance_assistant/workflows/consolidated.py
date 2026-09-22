"""Q3 — consolidated USD spend for a quarter. One of the two clean refusals.

REFUSED for the exact total whenever a required FX rate is missing —
"the total" is exactly the number a missing rate makes dishonest to state,
and there is no partial reading of "the exact total" that isn't misleading.
Computable components (by entity) and the non-convertible local amount are
still reported, just routed through `calculations` rather than `result`,
since the EvidenceBundle validator forbids `result` under REFUSED — `result`
stays reserved for "the exact answer to the question as literally asked",
which here genuinely is None.
"""

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import (
    AnswerStatus,
    CalcStep,
    Coverage,
    EvidenceBundle,
    MissingEvidence,
    MissingEvidenceReasonCode,
)
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.fx import aggregate_usd, aggregate_usd_by, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.workflows._shared import ToolTrace, quarter_bounds, resolve_year_or_readings


def consolidated_spend(
    gl: pd.DataFrame,
    fx: pd.DataFrame,
    quarter: str,
    year: int | None = None,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> EvidenceBundle:
    tt = ToolTrace()

    def compute(y: int):
        start, end = quarter_bounds(y, quarter)
        ledger = tt.call(query_ledger, gl, start, end, date_field=date_field)
        if ledger.rows_matched == 0:
            return None, None, {"has_data": False}

        fx_result = tt.call(convert_to_usd, ledger.rows, fx, date_field=date_field)
        total = tt.call(aggregate_usd, fx_result)
        by_entity = tt.call(aggregate_usd_by, fx_result, by=["entity"])

        status_hint = AnswerStatus.ANSWER if fx_result.coverage.is_complete else AnswerStatus.REFUSED
        payload = {"has_data": True, "fx_result": fx_result, "total": total, "by_entity": by_entity}
        return status_hint, total.converted_amount_usd, payload

    chosen_year, payload, period_readings = resolve_year_or_readings(gl, quarter, year, compute, date_field)

    if not payload["has_data"]:
        gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=(year is not None))
        return EvidenceBundle(
            status=gate_result.final_status,
            intent=Intent.CONSOLIDATED_SPEND,
            result=None,
            coverage=Coverage(selected_rows=0, computable_rows=0, computable_amount_pct=0.0),
            warnings=gate_result.warnings_added,
            refusal_reason=f"no ledger rows found for {quarter} {chosen_year}" if gate_result.final_status == AnswerStatus.REFUSED else None,
            clarification_options=gate_result.clarification_options,
            tool_calls=tt.calls,
        )

    fx_result = payload["fx_result"]
    total = payload["total"]
    by_entity = payload["by_entity"]
    coverage = Coverage.from_fx_coverage(fx_result.coverage)

    calculations: list[CalcStep] = [
        CalcStep(
            description="computable USD total (FX-convertible rows only)",
            operation="sum",
            inputs={"selected_rows": fx_result.coverage.selected_rows, "convertible_rows": fx_result.coverage.convertible_rows},
            output=total.converted_amount_usd,
        ),
        CalcStep(
            description="computable components by entity (USD)",
            operation="groupby_sum",
            inputs={"by": "entity"},
            output={key[0]: v.converted_amount_usd for key, v in by_entity.items()},
        ),
    ]

    missing_evidence: list[MissingEvidence] = []
    non_convertible_by_currency: dict[str, float] = {}
    for m in fx_result.missing:
        non_convertible_by_currency[m.currency] = non_convertible_by_currency.get(m.currency, 0.0) + m.affected_amount_local
        missing_evidence.append(
            MissingEvidence(
                what=f"FX rate for {m.currency} in {m.period_month}",
                reason=f"{m.affected_rows} row(s) totaling {m.affected_amount_local} in local currency could not be converted to USD",
                reason_code=MissingEvidenceReasonCode.MISSING_FX_RATE,
            )
        )
    if non_convertible_by_currency:
        calculations.append(
            CalcStep(
                description="non-convertible local amount by currency",
                operation="groupby_sum",
                inputs={"by": "currency", "source": "MissingFXRate.affected_amount_local"},
                output=non_convertible_by_currency,
            )
        )

    assumptions = [f"date basis: {date_field}"]

    gate_result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        fx_coverage=coverage,
        missing_fx_affects_requested_total=not fx_result.coverage.is_complete,
        force_refuse_incomplete_total=True,
        period_readings=period_readings,
    )

    # Only claim the "defaulted to most recent year" assumption when that
    # default is actually what the returned bundle relies on — a
    # NEEDS_CLARIFICATION outcome means the gate rejected the default.
    if year is None and gate_result.final_status != AnswerStatus.NEEDS_CLARIFICATION:
        assumptions.append(f"year not specified: defaulted to the most recent year with data, {chosen_year}")

    refusal_reason = None
    if gate_result.final_status == AnswerStatus.REFUSED:
        currencies = ", ".join(sorted(non_convertible_by_currency))
        refusal_reason = (
            f"exact consolidated USD total for {quarter} {chosen_year} cannot be stated: "
            f"{len(fx_result.missing)} currency/period combination(s) have no fx_rates.csv rate ({currencies}). "
            "Computable components and non-convertible local amounts are reported in calculations."
        )

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.CONSOLIDATED_SPEND,
        result={
            "quarter": quarter,
            "year": chosen_year,
            "date_basis": date_field,
            "exact_total_usd": total.converted_amount_usd,
        }
        if gate_result.final_status not in (AnswerStatus.REFUSED, AnswerStatus.NEEDS_CLARIFICATION)
        else None,
        assumptions=assumptions,
        warnings=gate_result.warnings_added,
        missing_evidence=missing_evidence,
        coverage=coverage,
        calculations=calculations,
        refusal_reason=refusal_reason,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
