"""Q5 — worst cost centres vs budget, with a driver established before any
document is searched.

Causal order (decided, not reinterpretable): the numeric driver account is
established from the ledger FIRST, for the worst-performing cost centre.
Only after that number exists does this workflow search the board memo —
and it searches for the driver account it just found, never a generic
"variance" query. If the memo's narrative doesn't actually discuss that
account, the mismatch is reported as a warning instead of silently
repeating the memo's narrative as though it explained the numbers. This is
R8's rule made concrete: the ledger establishes the fact, the document may
only explain it — never the reverse.
"""

from pathlib import Path

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
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
from finance_assistant.tools.accounts import resolve_account_hierarchy
from finance_assistant.tools.budget import query_budget
from finance_assistant.tools.cost_centres import normalize_reporting_cost_centre
from finance_assistant.tools.documents import search_documents, tokenize
from finance_assistant.tools.fx import aggregate_usd_by, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.workflows._shared import ToolTrace, quarter_bounds


def budget_variance(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    budget: pd.DataFrame,
    quarter: str,
    year: int,
    documents_dir: str | Path | None = None,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> EvidenceBundle:
    tt = ToolTrace()
    start, end = quarter_bounds(year, quarter)
    period_start, period_end = start.strftime("%Y-%m"), end.strftime("%Y-%m")

    budget_covers_period = budget["period_month"].between(period_start, period_end, inclusive="both").any()
    if not budget_covers_period:
        gate_result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=True)
        return EvidenceBundle(
            status=gate_result.final_status,
            intent=Intent.BUDGET_VARIANCE,
            result=None,
            coverage=Coverage(selected_rows=0, computable_rows=0, computable_amount_pct=0.0),
            missing_evidence=[
                MissingEvidence(
                    what=f"budget for {quarter} {year}",
                    reason=f"budget.csv has no period_month rows between {period_start} and {period_end}",
                    reason_code=MissingEvidenceReasonCode.NO_DATA_FOR_PERIOD,
                )
            ],
            warnings=gate_result.warnings_added,
            refusal_reason=f"budget.csv does not cover {quarter} {year} (R5: budget covers one year only)",
            clarification_options=gate_result.clarification_options,
            tool_calls=tt.calls,
        )

    # 1. Numeric driver established from the ledger FIRST, before any document search.
    ledger = tt.call(query_ledger, gl, start, end, date_field=date_field)
    hierarchy = tt.call(resolve_account_hierarchy, ledger.rows, coa, date_field=date_field, strict=False)
    normalized = tt.call(normalize_reporting_cost_centre, hierarchy.rows, date_field=date_field)
    fx_result = tt.call(convert_to_usd, normalized.rows, fx, date_field=date_field)
    actual_by_cc = tt.call(aggregate_usd_by, fx_result, by=["reporting_cost_centre"])
    actual_usd_by_cost_centre = {key[0]: v.converted_amount_usd for key, v in actual_by_cc.items()}

    budget_result = tt.call(
        query_budget,
        budget,
        period_start=period_start,
        period_end=period_end,
        actual_usd_by_cost_centre=actual_usd_by_cost_centre,
    )
    budget_by_cc = budget_result.aggregated_rows.groupby("cost_centre")["budget_amount"].sum().to_dict()

    variance_by_cc = {
        cc: actual_usd_by_cost_centre[cc] - budget_by_cc.get(cc, 0.0) for cc in actual_usd_by_cost_centre
    }
    # Adverse variance = spend over plan; rank most adverse first.
    adverse_ranking = sorted(variance_by_cc.items(), key=lambda kv: kv[1], reverse=True)

    warnings: list[str] = []
    for cc, _ in adverse_ranking:
        cc_coverage = actual_by_cc[(cc,)].coverage
        if cc_coverage.convertible_rows < cc_coverage.selected_rows:
            warnings.append(
                f"cost centre {cc!r} has incomplete FX coverage ({cc_coverage.convertible_rows}/"
                f"{cc_coverage.selected_rows} rows) — its actual spend is understated and it may appear "
                "artificially below plan"
            )

    if not adverse_ranking:
        worst_cc = None
        driver_account = None
        account_decomposition: dict[str, float] = {}
    else:
        worst_cc = adverse_ranking[0][0]
        worst_rows = normalized.rows.loc[normalized.rows["reporting_cost_centre"] == worst_cc]
        worst_fx = tt.call(convert_to_usd, worst_rows, fx, date_field=date_field)
        by_account = tt.call(aggregate_usd_by, worst_fx, by=["account_code", "account_name"])

        worst_budget = tt.call(query_budget, budget, period_start=period_start, period_end=period_end, cost_centres=[worst_cc])
        budget_by_account = worst_budget.aggregated_rows.groupby("account_code")["budget_amount"].sum().to_dict()

        account_decomposition = {
            key[0]: v.converted_amount_usd - budget_by_account.get(key[0], 0.0) for key, v in by_account.items()
        }
        account_names = {key[0]: key[1] for key in by_account}
        driver_code = max(account_decomposition, key=account_decomposition.get) if account_decomposition else None
        driver_account = (
            {
                "account_code": driver_code,
                "account_name": account_names.get(driver_code),
                "variance_usd": account_decomposition[driver_code],
                "memo_confirmation": None,
            }
            if driver_code
            else None
        )

    # 2. Only now, search the document — for the driver account just found,
    # never a generic query. Term overlap alone is not confirmation: a
    # section that shares one word with the driver account name (e.g. both
    # mention "freight") is not the same as a section that actually
    # discusses that specific account. Three explicit outcomes, generic
    # (set comparison only, no dataset word ever named in this code):
    #   CONFIRMED       — matched_terms cover the driver name's FULL term set
    #   PARTIAL_OVERLAP — some but not all terms matched
    #   NO_MATCH        — no section scored above zero at all
    sources: list[SourceRef] = []
    if driver_account and driver_account.get("account_name"):
        driver_name = str(driver_account["account_name"])
        driver_terms = set(tokenize(driver_name))
        search_result = tt.call(
            search_documents,
            query=driver_name, filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir, max_results=1
        )

        if not search_result.matches:
            confirmation_state = "NO_MATCH"
            warnings.append(
                f"board_memo_2024_q2.md has no section discussing {driver_name!r}, the account actually "
                f"driving the {worst_cc!r} variance — do not attribute this variance to any memo narrative "
                "not verified against this account"
            )
        else:
            best = search_result.matches[0]
            matched_terms = set(best.matched_terms)
            if driver_terms and driver_terms.issubset(matched_terms):
                confirmation_state = "CONFIRMED"
                sources.append(SourceRef.from_document_match(best))
            else:
                confirmation_state = "PARTIAL_OVERLAP"
                missing_terms = sorted(driver_terms - matched_terms)
                warnings.append(
                    f"board_memo_2024_q2.md section {best.section!r} only partially overlaps the driver "
                    f"account {driver_name!r} (matched term(s): {sorted(matched_terms)}; missing term(s): "
                    f"{missing_terms}) — the real driver for {worst_cc!r} by amount is {driver_name!r}, and "
                    "this section does not specifically identify that account. Do not treat partial term "
                    "overlap as confirmation of the narrative."
                )
        driver_account["memo_confirmation"] = confirmation_state

    relevant_ambiguous = any(
        pc.is_ambiguous for pc in budget_result.plausibility_checks if pc.affected_cost_centre in variance_by_cc
    )

    row_coverage = Coverage(
        selected_rows=hierarchy.total_rows,
        computable_rows=hierarchy.matched_rows,
        computable_amount_pct=(hierarchy.matched_rows / hierarchy.total_rows * 100.0) if hierarchy.total_rows else 100.0,
    )
    fx_coverage = Coverage.from_fx_coverage(fx_result.coverage)

    assumptions = [
        f"date basis: {date_field}",
        "R4 reporting_cost_centre normalization applied before comparing actual to budget",
        f"budget duplicate-key aggregation rule: {budget_result.aggregation_rule}",
    ]
    if budget_result.plausibility_checks:
        assumptions.append(
            f"{len(budget_result.plausibility_checks)} cost centre(s) had duplicate budget keys; "
            f"plausibility check results: {[(pc.affected_cost_centre, pc.preferred_hypothesis, pc.is_ambiguous) for pc in budget_result.plausibility_checks]}"
        )

    if budget_result.duplicate_keys:
        warnings.append(
            f"{len(budget_result.duplicate_keys)} duplicate budget key(s) detected, affecting cost centre(s) "
            f"{sorted({d.cost_centre for d in budget_result.duplicate_keys})}"
        )

    result = {
        "quarter": quarter,
        "year": year,
        "date_basis": date_field,
        "variance_ranking_usd": [{"cost_centre": cc, "variance_usd": v} for cc, v in adverse_ranking],
        "worst_cost_centre": worst_cc,
        "worst_cost_centre_account_decomposition_usd": account_decomposition,
        "driver_account": driver_account,
        "duplicate_budget_keys": len(budget_result.duplicate_keys),
        "budget_aggregation_rule": budget_result.aggregation_rule,
        "plausibility_checks": [
            {
                "affected_cost_centre": pc.affected_cost_centre,
                "preferred_hypothesis": pc.preferred_hypothesis,
                "is_ambiguous": pc.is_ambiguous,
            }
            for pc in budget_result.plausibility_checks
        ],
    }

    gate_result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        fx_coverage=fx_coverage,
        missing_fx_affects_requested_total=not fx_result.coverage.is_complete,
        row_coverage=row_coverage,
        grouping_would_change_result=relevant_ambiguous,
    )

    missing_evidence: list[MissingEvidence] = []
    for m in fx_result.missing:
        missing_evidence.append(
            MissingEvidence(
                what=f"FX rate for {m.currency} in {m.period_month}",
                reason=f"{m.affected_rows} row(s) totaling {m.affected_amount_local} in local currency could not be converted to USD",
                reason_code=MissingEvidenceReasonCode.MISSING_FX_RATE,
            )
        )

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.BUDGET_VARIANCE,
        result=result if gate_result.final_status not in (AnswerStatus.REFUSED, AnswerStatus.NEEDS_CLARIFICATION) else None,
        sources=sources,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        missing_evidence=missing_evidence,
        coverage=fx_coverage,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
