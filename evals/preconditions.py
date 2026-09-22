"""Independently-computed preconditions for the `conditional` eval cases.

Each function here establishes a ground-truth fact using only
`finance_assistant.tools.*` (plus, for the opex case, the gate's own
declared ambiguity function/constant) — never by calling the workflow
under test, and never by reimplementing a gate decision threshold from
scratch. This mirrors docs/PROMPT_MAESTRO.md's own testing principle:
expected values are established independently, never by invoking the same
function under test.

A workflow call and a precondition call may legitimately compose the same
low-level tools (query_ledger, convert_to_usd, ...) — that is not circular.
What would be circular is deriving the precondition from the *workflow's*
own output status, which none of these do.
"""

from dataclasses import dataclass

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD, OPEX_STATEMENT_LINE
from finance_assistant.evidence.gate import DEFAULT_PERIOD_READING_TOLERANCE, readings_are_material_difference
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.tools.accounts import resolve_account_hierarchy
from finance_assistant.tools.budget import query_budget
from finance_assistant.tools.cost_centres import normalize_reporting_cost_centre
from finance_assistant.tools.fx import aggregate_usd, aggregate_usd_by, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.tools.vendors import detect_alias_clusters, normalize_vendor_name, vendor_lookup
from finance_assistant.workflows._shared import quarter_bounds


@dataclass(frozen=True)
class PreconditionResult:
    holds: bool
    detail: str
    # Populated only by opex_q2_period_ambiguous: the exact period-reading
    # labels (e.g. "FY2024 Q2") it independently computed, in the same
    # format apply_gate's row D builds clarification_options from. Lets the
    # runner assert NEEDS_CLARIFICATION actually names these years, without
    # hardcoding a dataset-specific year anywhere in questions.yaml.
    labels: tuple[str, ...] = ()


def _years_present(gl: pd.DataFrame, date_field: str) -> list[int]:
    return sorted({int(y) for y in gl[date_field].dropna().dt.year.unique()})


def opex_q2_period_ambiguous(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    quarter: str,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> PreconditionResult:
    """Independently computes each year's opex-perimeter USD total for
    `quarter` (via tools, mirroring workflows/opex.py's own per-year
    computation without calling it), then asks the gate's own declared
    materiality function whether those readings differ enough to force
    NEEDS_CLARIFICATION. Imports DEFAULT_PERIOD_READING_TOLERANCE rather
    than guessing or re-deriving a threshold."""

    readings: list[tuple[str, AnswerStatus | None, float | None]] = []
    for year in _years_present(gl, date_field):
        start, end = quarter_bounds(year, quarter)
        ledger = query_ledger(gl, start, end, date_field=date_field)
        if ledger.rows_matched == 0:
            continue
        hierarchy = resolve_account_hierarchy(ledger.rows, coa, date_field=date_field, strict=False)
        perimeter_rows = hierarchy.rows.loc[hierarchy.rows["statement_line"] == OPEX_STATEMENT_LINE]
        normalized = normalize_reporting_cost_centre(perimeter_rows, date_field=date_field)
        fx_result = convert_to_usd(normalized.rows, fx, date_field=date_field)
        total = aggregate_usd(fx_result).converted_amount_usd
        readings.append((f"FY{year} {quarter}", AnswerStatus.ANSWER, total))

    labels = tuple(label for label, _, _ in readings)
    readable_readings = "; ".join(f"{label} = ${amount:,.2f}" for label, _, amount in readings) or "none"

    if len(readings) < 2:
        return PreconditionResult(
            holds=False,
            detail=f"only {len(readings)} year(s) with {quarter} opex data ({readable_readings}); ambiguity requires at least 2",
            labels=labels,
        )

    holds = readings_are_material_difference(readings, tolerance=DEFAULT_PERIOD_READING_TOLERANCE)
    return PreconditionResult(
        holds=holds,
        detail=(
            f"{readable_readings} (tolerance {DEFAULT_PERIOD_READING_TOLERANCE:.0%}, "
            f"evidence.gate.DEFAULT_PERIOD_READING_TOLERANCE) -> materially different: {holds}"
        ),
        labels=labels,
    )


def missing_fx_rate_in_period(
    gl: pd.DataFrame,
    fx: pd.DataFrame,
    quarter: str,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> PreconditionResult:
    """Resolves the same "most recent year with data for this quarter"
    year that consolidated_spend(..., year=None) would settle on, then
    checks FX completeness for that exact period via convert_to_usd
    directly — never via the workflow."""

    years = _years_present(gl, date_field)
    chosen_year = None
    ledger_rows = None
    for year in reversed(years):
        start, end = quarter_bounds(year, quarter)
        ledger = query_ledger(gl, start, end, date_field=date_field)
        if ledger.rows_matched > 0:
            chosen_year, ledger_rows = year, ledger.rows
            break

    if chosen_year is None:
        return PreconditionResult(holds=False, detail=f"no ledger rows found for {quarter} in any year present {years}")

    fx_result = convert_to_usd(ledger_rows, fx, date_field=date_field)
    holds = len(fx_result.missing) > 0
    if holds:
        readable_missing = "; ".join(
            f"missing FX rate for {m.currency} in {m.period_month} ({m.affected_rows} row(s) affected)"
            for m in fx_result.missing
        )
    else:
        readable_missing = "none"
    detail = f"{quarter} {chosen_year}: {readable_missing}"
    return PreconditionResult(holds=holds, detail=detail)


def coa_reclassification_between_years(coa: pd.DataFrame, year_prior: int, year_current: int) -> PreconditionResult:
    """Does any account_code have more than one chart-of-accounts row whose
    valid_from falls strictly inside [Jan 1 year_prior, Dec 31
    year_current]? A pure structural read of chart_of_accounts.csv, not a
    replication of workflows/travel.py's T&E-specific bridge/sign logic."""

    window_start = pd.Timestamp(year=year_prior, month=1, day=1)
    window_end = pd.Timestamp(year=year_current, month=12, day=31)

    changed_codes: list[str] = []
    for account_code, group in coa.groupby("account_code"):
        if len(group) < 2:
            continue
        changed_in_window = group["valid_from"].between(window_start, window_end, inclusive="both")
        if changed_in_window.any():
            changed_codes.append(account_code)

    holds = len(changed_codes) > 0
    readable_codes = ", ".join(sorted(changed_codes)) if changed_codes else "none"
    return PreconditionResult(
        holds=holds,
        detail=f"account(s) reclassified within {window_start.date()}–{window_end.date()}: {readable_codes}",
    )


def alias_clusters_change_topn(
    gl: pd.DataFrame,
    vendors: pd.DataFrame,
    fx: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    top_n: int,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> PreconditionResult:
    """Independently composes query_ledger + vendor_lookup + convert_to_usd
    + aggregate_usd_by + detect_alias_clusters (all tools) to build both
    top-N vendor sets and compare — mirroring workflows/vendors.py's own
    computation via tools composition, not by calling it."""

    ledger = query_ledger(gl, date_start, date_end, date_field=date_field)
    lookup = vendor_lookup(ledger.rows, vendors)
    fx_result = convert_to_usd(lookup.rows, fx, date_field=date_field)

    has_vendor_id = fx_result.rows["vendor_id"].notna()
    vendor_id_rows = fx_result.rows.loc[has_vendor_id]
    vendor_id_fx = fx_result.__class__(rows=vendor_id_rows, coverage=fx_result.coverage, missing=fx_result.missing)
    ranking_by_vendor_id = aggregate_usd_by(vendor_id_fx, by=["vendor_id"])

    alias_clusters = detect_alias_clusters(vendors)
    normalized_by_vendor_id = dict(zip(vendors["vendor_id"], vendors["vendor_name"].map(normalize_vendor_name)))
    cluster_by_vendor_id: dict[str, str] = {}
    for cluster in alias_clusters.clusters:
        for vid in cluster.vendor_ids:
            cluster_by_vendor_id[vid] = cluster.normalized_name

    working = vendor_id_rows.copy()
    working["_cluster_key"] = working["vendor_id"].map(
        lambda v: cluster_by_vendor_id.get(v, normalized_by_vendor_id.get(v, v))
    )
    cluster_fx = fx_result.__class__(rows=working, coverage=fx_result.coverage, missing=fx_result.missing)
    ranking_by_cluster = aggregate_usd_by(cluster_fx, by=["_cluster_key"])

    def top_vendor_ids(ranking: dict, key_to_vendor_ids: dict[str, set[str]] | None = None) -> set[str]:
        ranked = sorted(ranking.items(), key=lambda kv: kv[1].converted_amount_usd, reverse=True)[:top_n]
        if key_to_vendor_ids is None:
            return {key[0] for key, _ in ranked}
        result: set[str] = set()
        for (key,), _ in ranked:
            result |= key_to_vendor_ids.get(key, set())
        return result

    cluster_key_to_vendor_ids: dict[str, set[str]] = {}
    for vid, key in zip(working["vendor_id"], working["_cluster_key"]):
        cluster_key_to_vendor_ids.setdefault(key, set()).add(vid)

    top_by_vendor_id = top_vendor_ids(ranking_by_vendor_id)
    top_by_cluster = top_vendor_ids(ranking_by_cluster, cluster_key_to_vendor_ids)

    holds = top_by_vendor_id != top_by_cluster
    readable_by_vendor_id = ", ".join(sorted(top_by_vendor_id))
    readable_by_cluster = ", ".join(sorted(top_by_cluster))
    return PreconditionResult(
        holds=holds,
        detail=f"top-{top_n} by vendor_id: {readable_by_vendor_id} vs by alias cluster: {readable_by_cluster}",
    )


def fx_incomplete_or_budget_keys_ambiguous(
    gl: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    budget: pd.DataFrame,
    quarter: str,
    year: int,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> PreconditionResult:
    """ORs two independently-computed facts: (a) FX incompleteness for the
    period, (b) an ambiguous duplicate budget key (per tools/budget.py's
    own plausibility check), computed by composing query_ledger +
    resolve_account_hierarchy + normalize_reporting_cost_centre +
    convert_to_usd + aggregate_usd_by + query_budget — mirroring
    workflows/variance.py step 1 via tools composition, not by calling it."""

    start, end = quarter_bounds(year, quarter)
    period_start, period_end = start.strftime("%Y-%m"), end.strftime("%Y-%m")

    ledger = query_ledger(gl, start, end, date_field=date_field)
    hierarchy = resolve_account_hierarchy(ledger.rows, coa, date_field=date_field, strict=False)
    normalized = normalize_reporting_cost_centre(hierarchy.rows, date_field=date_field)
    fx_result = convert_to_usd(normalized.rows, fx, date_field=date_field)
    fx_incomplete = len(fx_result.missing) > 0

    actual_by_cc = aggregate_usd_by(fx_result, by=["reporting_cost_centre"])
    actual_usd_by_cost_centre = {key[0]: v.converted_amount_usd for key, v in actual_by_cc.items()}

    budget_result = query_budget(
        budget,
        period_start=period_start,
        period_end=period_end,
        actual_usd_by_cost_centre=actual_usd_by_cost_centre,
    )
    ambiguous_ccs = [pc.affected_cost_centre for pc in budget_result.plausibility_checks if pc.is_ambiguous]

    holds = fx_incomplete or bool(ambiguous_ccs)
    if fx_incomplete:
        readable_missing = "; ".join(f"{m.currency} in {m.period_month}" for m in fx_result.missing)
        fx_part = f"FX incomplete: missing rate(s) for {readable_missing}"
    else:
        fx_part = "FX complete"
    budget_part = (
        f"ambiguous budget cost centre(s): {', '.join(ambiguous_ccs)}"
        if ambiguous_ccs
        else "no ambiguous budget cost centres"
    )
    detail = f"{fx_part}; {budget_part}"
    return PreconditionResult(holds=holds, detail=detail)
