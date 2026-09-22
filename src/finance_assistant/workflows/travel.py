"""Q2 — travel 2024 vs 2023, reported basis vs TWO comparable bases.

Always PARTIAL: comparing T&E across a year in which client entertainment
was reclassified out of the "Travel & Entertainment" COA parent is a
non-authoritative grouping decision (which basis you read the comparison
on), not a data-quality gap.

Fixing a single comparable reference date would hide one of two equally
legitimate readings: "how does travel compare under today's definition?"
(reference = the most recent date in the combined row set) vs "how does it
compare under the definition in force at the start of the period?"
(reference = the earliest date). Both are computed, both are returned, and
the reported-vs-comparable sign check runs against each of them
independently — collapsing to one would silently make the choice for the
reader.
"""

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import (
    AnswerStatus,
    CalcStep,
    Coverage,
    EvidenceBundle,
    SourceRef,
)
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.accounts import resolve_account_hierarchy
from finance_assistant.tools.documents import search_documents
from finance_assistant.tools.fx import aggregate_usd, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.tools.te_policy import load_policy_rules
from finance_assistant.workflows._shared import ToolTrace


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)


def _te_total(rows: pd.DataFrame, coa: pd.DataFrame, fx: pd.DataFrame, date_field: str, parent_name: str, tt: ToolTrace):
    hierarchy = tt.call(resolve_account_hierarchy, rows, coa, date_field=date_field, strict=False)
    te_rows = hierarchy.rows.loc[hierarchy.rows["parent_name"] == parent_name]
    fx_result = tt.call(convert_to_usd, te_rows, fx, date_field=date_field)
    total = tt.call(aggregate_usd, fx_result)
    account_codes = sorted(te_rows["account_code"].dropna().unique().tolist())
    return te_rows, fx_result, total, account_codes


def travel_comparison(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    year_current: int,
    year_prior: int,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> EvidenceBundle:
    tt = ToolTrace()
    parent_name = tt.call(load_policy_rules).te_perimeter.parent_name

    years = {"current": year_current, "prior": year_prior}
    ledgers: dict[str, pd.DataFrame] = {}
    for key, y in years.items():
        start, end = _year_bounds(y)
        ledgers[key] = tt.call(query_ledger, gl, start, end, date_field=date_field).rows

    combined = pd.concat(ledgers.values())
    reference_date_current = combined[date_field].max()
    reference_date_prior = combined[date_field].min()

    reported = {}
    calculations: list[CalcStep] = []
    coverages: list[Coverage] = []
    for key, y in years.items():
        rows = ledgers[key]
        te_rows, fx_result, total, codes = _te_total(rows, coa, fx, date_field, parent_name, tt)
        reported[key] = {"rows": te_rows, "fx_result": fx_result, "total": total, "account_codes": codes}
        coverages.append(Coverage.from_fx_coverage(fx_result.coverage))

    comparable_bases: dict[str, dict] = {}
    bridge_usd: dict[str, dict[str, float]] = {"current": {}, "prior": {}}
    reclassified_codes: dict[str, list[str]] = {"current": [], "prior": []}

    for basis_name, ref_date in (("current", reference_date_current), ("prior", reference_date_prior)):
        comparable_bases[basis_name] = {}
        all_reclassified: set[str] = set()
        for key, y in years.items():
            rows = ledgers[key].copy()
            ref_col = f"_comparable_ref_date_{basis_name}"
            rows[ref_col] = ref_date
            te_rows, fx_result, total, codes = _te_total(rows, coa, fx, ref_col, parent_name, tt)
            comparable_bases[basis_name][key] = {
                "rows": te_rows,
                "fx_result": fx_result,
                "total": total,
                "account_codes": codes,
            }
            coverages.append(Coverage.from_fx_coverage(fx_result.coverage))

            reported_codes = set(reported[key]["account_codes"])
            comparable_codes = set(codes)
            changed = reported_codes.symmetric_difference(comparable_codes)
            all_reclassified |= changed

            reported_idx = set(reported[key]["rows"].index)
            comparable_idx = set(te_rows.index)
            rows_added_idx = comparable_idx - reported_idx
            rows_removed_idx = reported_idx - comparable_idx
            added_usd = tt.call(convert_to_usd, ledgers[key].loc[list(rows_added_idx)], fx, date_field=date_field)
            removed_usd = tt.call(convert_to_usd, ledgers[key].loc[list(rows_removed_idx)], fx, date_field=date_field)
            added_total = tt.call(aggregate_usd, added_usd).converted_amount_usd
            removed_total = tt.call(aggregate_usd, removed_usd).converted_amount_usd
            bridge = added_total - removed_total
            bridge_usd[basis_name][key] = bridge
            calculations.append(
                CalcStep(
                    description=f"reclassification bridge, {basis_name} basis, FY{y}",
                    operation="net_reclassified_amount",
                    inputs={
                        "reference_date": str(ref_date),
                        "rows_added": len(rows_added_idx),
                        "rows_removed": len(rows_removed_idx),
                    },
                    output=bridge,
                )
            )
        reclassified_codes[basis_name] = sorted(all_reclassified)

    def variance(a: float, b: float) -> float:
        return a - b

    reported_variance = variance(
        reported["current"]["total"].converted_amount_usd, reported["prior"]["total"].converted_amount_usd
    )
    comparable_variance = {
        basis: variance(
            comparable_bases[basis]["current"]["total"].converted_amount_usd,
            comparable_bases[basis]["prior"]["total"].converted_amount_usd,
        )
        for basis in ("current", "prior")
    }

    def sign(x: float) -> int:
        return (x > 0) - (x < 0)

    sign_mismatch = {basis: sign(reported_variance) != sign(comparable_variance[basis]) for basis in ("current", "prior")}

    warnings: list[str] = []
    for basis in ("current", "prior"):
        if sign_mismatch[basis]:
            warnings.append(
                f"variance sign differs between reported basis ({reported_variance:+.2f}) and the "
                f"{basis} comparable basis ({comparable_variance[basis]:+.2f}) — reading only one of "
                "these would misstate the direction of the change"
            )

    grouping_would_change_result = (
        any(bridge_usd["current"][k] != 0 for k in years)
        or any(bridge_usd["prior"][k] != 0 for k in years)
        or sign_mismatch["current"]
        or sign_mismatch["prior"]
    )

    reclassified_account_name = None
    for basis in ("current", "prior"):
        if reclassified_codes[basis]:
            code = reclassified_codes[basis][0]
            match = coa.loc[coa["account_code"] == code, "account_name"]
            if not match.empty:
                reclassified_account_name = match.iloc[0]
            break

    sources: list[SourceRef] = []
    if reclassified_account_name:
        search_result = tt.call(
            search_documents,
            query=f"{reclassified_account_name} reclassification",
            filenames=["board_memo_2024_q2.md"],
            max_results=1,
        )
        if search_result.matches:
            sources.append(SourceRef.from_document_match(search_result.matches[0]))

    coverage = Coverage.combine(*coverages)

    assumptions = [
        f"date basis: {date_field}",
        f"T&E perimeter: chart_of_accounts.parent_name == {parent_name!r} (source: travel_expense_policy.md, section 'Scope')",
        f"comparable basis 'current' reference date: {reference_date_current} (max of combined date range)",
        f"comparable basis 'prior' reference date: {reference_date_prior} (min of combined date range)",
    ]

    def _basis_result(entry: dict) -> dict:
        return {
            "te_total_usd": entry["total"].converted_amount_usd,
            "coverage": Coverage.from_fx_coverage(entry["fx_result"].coverage).model_dump(),
        }

    result = {
        "years": {"current": year_current, "prior": year_prior},
        "date_basis": date_field,
        "reported_basis": {
            "current": _basis_result(reported["current"]),
            "prior": _basis_result(reported["prior"]),
            "variance_usd": reported_variance,
        },
        "comparable_basis_current": {
            "current": _basis_result(comparable_bases["current"]["current"]),
            "prior": _basis_result(comparable_bases["current"]["prior"]),
            "variance_usd": comparable_variance["current"],
            "reference_date": str(reference_date_current),
        },
        "comparable_basis_prior": {
            "current": _basis_result(comparable_bases["prior"]["current"]),
            "prior": _basis_result(comparable_bases["prior"]["prior"]),
            "variance_usd": comparable_variance["prior"],
            "reference_date": str(reference_date_prior),
        },
        "reclassification_bridge_usd": bridge_usd,
        "reclassified_account_codes": reclassified_codes,
    }

    gate_result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        grouping_would_change_result=grouping_would_change_result,
        fx_coverage=coverage,
        missing_fx_affects_requested_total=not coverage.computable_amount_pct == 100.0,
    )

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.TRAVEL_COMPARISON,
        result=result if gate_result.final_status not in (AnswerStatus.REFUSED, AnswerStatus.NEEDS_CLARIFICATION) else None,
        sources=sources,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        coverage=coverage,
        calculations=calculations,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
