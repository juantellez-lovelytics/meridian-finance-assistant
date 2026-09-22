"""Q1 — opex by cost centre.

ANSWER (or NEEDS_CLARIFICATION on an omitted, ambiguous year). Declares the
date basis (accrual_date by default) and the opex perimeter
(config.OPEX_STATEMENT_LINE) explicitly, and reports how many rows that
perimeter filter actually excluded — including the zero case, since a
filter that silently excludes nothing is indistinguishable from a broken
one.
"""

import pandas as pd

from finance_assistant import config
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
from finance_assistant.tools.accounts import resolve_account_hierarchy
from finance_assistant.tools.cost_centres import normalize_reporting_cost_centre
from finance_assistant.tools.fx import aggregate_usd, aggregate_usd_by, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.workflows._shared import ToolTrace, quarter_bounds, resolve_year_or_readings


def opex_by_cost_centre(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
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

        hierarchy = tt.call(resolve_account_hierarchy, ledger.rows, coa, date_field=date_field, strict=False)
        total_rows_before_perimeter_filter = len(hierarchy.rows)
        perimeter_rows = hierarchy.rows.loc[hierarchy.rows["statement_line"] == config.OPEX_STATEMENT_LINE]

        normalized = tt.call(normalize_reporting_cost_centre, perimeter_rows, date_field=date_field)
        fx_result = tt.call(convert_to_usd, normalized.rows, fx, date_field=date_field)
        total = tt.call(aggregate_usd, fx_result)
        by_cc = tt.call(aggregate_usd_by, fx_result, by=["reporting_cost_centre"])

        status_hint = AnswerStatus.ANSWER if fx_result.coverage.is_complete else AnswerStatus.PARTIAL
        payload = {
            "has_data": True,
            "hierarchy": hierarchy,
            "total_rows_before_perimeter_filter": total_rows_before_perimeter_filter,
            "opex_perimeter_rows": len(perimeter_rows),
            "normalized": normalized,
            "fx_result": fx_result,
            "total": total,
            "by_cc": by_cc,
        }
        return status_hint, total.converted_amount_usd, payload

    chosen_year, payload, period_readings = resolve_year_or_readings(gl, quarter, year, compute, date_field)

    if not payload["has_data"]:
        gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=(year is not None))
        return EvidenceBundle(
            status=gate_result.final_status,
            intent=Intent.OPEX_BY_COST_CENTRE,
            result=None,
            coverage=Coverage(selected_rows=0, computable_rows=0, computable_amount_pct=0.0),
            warnings=gate_result.warnings_added,
            refusal_reason=f"no ledger rows found for {quarter} {chosen_year}" if gate_result.final_status == AnswerStatus.REFUSED else None,
            clarification_options=gate_result.clarification_options,
            tool_calls=tt.calls,
        )

    fx_result = payload["fx_result"]
    total = payload["total"]
    by_cc = payload["by_cc"]
    hierarchy = payload["hierarchy"]
    total_before = payload["total_rows_before_perimeter_filter"]
    perimeter_rows = payload["opex_perimeter_rows"]

    coverage = Coverage.from_fx_coverage(fx_result.coverage)

    assumptions = [
        f"date basis: {date_field}",
        f"opex perimeter: chart_of_accounts.statement_line == {config.OPEX_STATEMENT_LINE!r}",
    ]
    if perimeter_rows == total_before:
        assumptions.append("the perimeter did not exclude any row: every chart-of-accounts account belongs to the declared line")
    else:
        assumptions.append(f"the perimeter excluded {total_before - perimeter_rows} row(s) outside {config.OPEX_STATEMENT_LINE!r}")
    assumptions.append("R4 reporting_cost_centre normalization applied before aggregation")

    warnings: list[str] = []
    if hierarchy.unmapped:
        warnings.append(f"{len(hierarchy.unmapped)} row(s) had an unmapped account_code and were excluded from the opex perimeter")

    missing_evidence = []
    for m in fx_result.missing:
        missing_evidence.append(
            MissingEvidence(
                what=f"FX rate for {m.currency} in {m.period_month}",
                reason=f"{m.affected_rows} row(s) totaling {m.affected_amount_local} in local currency could not be converted to USD",
                reason_code=MissingEvidenceReasonCode.MISSING_FX_RATE,
            )
        )

    result = {
        "quarter": quarter,
        "year": chosen_year,
        "date_basis": date_field,
        "opex_perimeter": {"statement_line": config.OPEX_STATEMENT_LINE},
        "opex_perimeter_rows": perimeter_rows,
        "total_rows_before_perimeter_filter": total_before,
        "total_usd": total.converted_amount_usd,
        "by_cost_centre_usd": {key[0]: v.converted_amount_usd for key, v in by_cc.items()},
        "unmapped_account_rows": len(hierarchy.unmapped),
    }

    gate_result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        fx_coverage=coverage,
        missing_fx_affects_requested_total=not fx_result.coverage.is_complete,
        period_readings=period_readings,
    )

    # Only claim the "defaulted to most recent year" assumption when that
    # default is actually what the returned bundle relies on — a
    # NEEDS_CLARIFICATION outcome means the gate rejected the default, so
    # asserting it here would misrepresent what the bundle actually did.
    if year is None and gate_result.final_status != AnswerStatus.NEEDS_CLARIFICATION:
        assumptions.append(f"year not specified: defaulted to the most recent year with data, {chosen_year}")

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.OPEX_BY_COST_CENTRE,
        result=result if gate_result.final_status not in (AnswerStatus.REFUSED, AnswerStatus.NEEDS_CLARIFICATION) else None,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        missing_evidence=missing_evidence,
        coverage=coverage,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
