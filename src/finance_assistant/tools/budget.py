"""R5 — budget query with declared, non-destructive duplicate handling.

budget.csv carries repeated dimensional keys (same entity + cost_centre +
account_code + period_month, different budget_amount) wherever a cost
centre's plan was restated under a new reporting code for the full year
(R4): the restated rows and the centre's own native rows land on the same
key. `drop_duplicates()` or picking one row at random is prohibited —
this module detects the repeated keys generically (whichever cost centre
actually has them, never assumed), reports them as a diagnostic, and
aggregates them under one declared rule: additive (sum), because the
default reading is that repeated rows are components of the restated
plan, not competing alternative versions of it.

That reading is not just asserted: `query_budget` can also run the
plausibility check R5 describes. Given the caller's already-computed
actual USD spend per cost centre, it compares the affected cost centre's
actual/budget ratio under the additive hypothesis against the same ratio
under a "single row" hypothesis (one row per key, duplicates discarded)
and checks which one lands closer to the median actual/budget ratio of
the *other* cost centres. If both hypotheses land about equally close,
neither is decisively more plausible, and the check reports that
ambiguity instead of silently preferring the default.
"""

from dataclasses import dataclass
from typing import Hashable

import pandas as pd

_DIMENSION_KEY = ["entity", "cost_centre", "account_code", "period_month"]
_DIMENSION_COLUMNS = {"entities": "entity", "cost_centres": "cost_centre", "account_codes": "account_code"}

BUDGET_DUPLICATE_AGGREGATION_RULE = "additive"

# Declared tolerance for the R5 plausibility check: if the additive and
# single-row hypotheses land within this fraction of the peer median of
# each other's distance, neither is decisively more plausible and the
# check must say so rather than silently pick a winner.
PLAUSIBILITY_AMBIGUITY_TOLERANCE = 0.05


@dataclass(frozen=True)
class BudgetQueryFilters:
    period_start: str
    period_end: str
    entities: list[str] | None
    cost_centres: list[str] | None
    account_codes: list[str] | None


@dataclass(frozen=True)
class DuplicateBudgetKey:
    entity: str
    cost_centre: str
    account_code: str
    period_month: str
    row_count: int
    total_budget_amount: float
    row_indices: list[Hashable]


@dataclass(frozen=True)
class BudgetPlausibilityCheck:
    affected_cost_centre: str
    additive_ratio: float
    single_row_ratio: float
    peer_median_ratio: float
    peer_cost_centres: list[str]
    preferred_hypothesis: str  # "additive" | "single_row"
    is_ambiguous: bool


@dataclass(frozen=True)
class BudgetQueryResult:
    rows: pd.DataFrame  # raw filtered rows, untouched — duplicates are never dropped here
    aggregated_rows: pd.DataFrame  # one row per dimensional key, BUDGET_DUPLICATE_AGGREGATION_RULE applied
    filters: BudgetQueryFilters
    aggregation_rule: str
    duplicate_keys: list[DuplicateBudgetKey]
    plausibility_checks: list[BudgetPlausibilityCheck]


def query_budget(
    rows: pd.DataFrame,
    period_start: str,
    period_end: str,
    entities: list[str] | None = None,
    cost_centres: list[str] | None = None,
    account_codes: list[str] | None = None,
    actual_usd_by_cost_centre: dict[str, float] | None = None,
) -> BudgetQueryResult:
    """actual_usd_by_cost_centre, when supplied, must already be resolved to
    the same reporting_cost_centre basis (R4) and the same period as this
    query — that resolution is the caller's job (ledger + fx + cost_centres
    tools); this module stays budget-only and does not import them."""

    required_columns = ["entity", "cost_centre", "account_code", "period_month", "budget_amount", "currency"]
    missing_columns = [c for c in required_columns if c not in rows.columns]
    if missing_columns:
        raise ValueError(f"rows is missing required column(s): {', '.join(missing_columns)}")
    if period_start > period_end:
        raise ValueError(f"period_start ({period_start!r}) is after period_end ({period_end!r})")

    non_usd = rows.loc[rows["currency"] != "USD", "currency"].dropna().unique()
    if len(non_usd) > 0:
        raise ValueError(f"budget rows must be USD-denominated; found currency value(s): {list(non_usd)}")

    dimension_values = {"entities": entities, "cost_centres": cost_centres, "account_codes": account_codes}
    mask = rows["period_month"].between(period_start, period_end, inclusive="both")
    for dimension, values in dimension_values.items():
        if values is None:
            continue
        column = _DIMENSION_COLUMNS[dimension]
        mask &= rows[column].isin(values)

    filtered = rows.loc[mask]

    working = filtered.copy()
    working["_row_index"] = working.index
    grouped = working.groupby(_DIMENSION_KEY, dropna=False)
    aggregated = grouped.agg(
        budget_amount=("budget_amount", "sum"),
        row_count=("budget_amount", "size"),
        row_indices=("_row_index", list),
    ).reset_index()
    aggregated["currency"] = "USD"

    dup_groups = aggregated.loc[aggregated["row_count"] > 1]
    duplicate_keys = [
        DuplicateBudgetKey(
            entity=record["entity"],
            cost_centre=record["cost_centre"],
            account_code=record["account_code"],
            period_month=record["period_month"],
            row_count=int(record["row_count"]),
            total_budget_amount=float(record["budget_amount"]),
            row_indices=list(record["row_indices"]),
        )
        for record in dup_groups.to_dict("records")
    ]

    affected_cost_centres = sorted({d.cost_centre for d in duplicate_keys})

    plausibility_checks: list[BudgetPlausibilityCheck] = []
    if actual_usd_by_cost_centre and affected_cost_centres:
        single_row = working.sort_values("_row_index").groupby(_DIMENSION_KEY, dropna=False, as_index=False).first()
        additive_by_cc = aggregated.groupby("cost_centre")["budget_amount"].sum()
        single_by_cc = single_row.groupby("cost_centre")["budget_amount"].sum()

        for affected in affected_cost_centres:
            if affected not in actual_usd_by_cost_centre:
                continue
            actual = actual_usd_by_cost_centre[affected]
            additive_budget = additive_by_cc.get(affected)
            single_budget = single_by_cc.get(affected)
            if not additive_budget or not single_budget:
                continue

            peer_ratios = {
                cc: float(actual_usd_by_cost_centre[cc] / cc_budget)
                for cc, cc_budget in additive_by_cc.items()
                if cc != affected and cc in actual_usd_by_cost_centre and cc_budget
            }
            if not peer_ratios:
                continue

            peer_median = float(pd.Series(list(peer_ratios.values())).median())
            additive_ratio = float(actual / additive_budget)
            single_ratio = float(actual / single_budget)
            additive_distance = abs(additive_ratio - peer_median)
            single_distance = abs(single_ratio - peer_median)
            spread = max(additive_distance, single_distance)

            is_ambiguous = bool(
                spread == 0 or (abs(additive_distance - single_distance) / spread) <= PLAUSIBILITY_AMBIGUITY_TOLERANCE
            )
            preferred = "additive" if additive_distance <= single_distance else "single_row"

            plausibility_checks.append(
                BudgetPlausibilityCheck(
                    affected_cost_centre=affected,
                    additive_ratio=additive_ratio,
                    single_row_ratio=single_ratio,
                    peer_median_ratio=peer_median,
                    peer_cost_centres=sorted(peer_ratios.keys()),
                    preferred_hypothesis=preferred,
                    is_ambiguous=is_ambiguous,
                )
            )

    filters = BudgetQueryFilters(
        period_start=period_start,
        period_end=period_end,
        entities=entities,
        cost_centres=cost_centres,
        account_codes=account_codes,
    )

    return BudgetQueryResult(
        rows=filtered,
        aggregated_rows=aggregated.drop(columns=["row_indices"]),
        filters=filters,
        aggregation_rule=BUDGET_DUPLICATE_AGGREGATION_RULE,
        duplicate_keys=duplicate_keys,
        plausibility_checks=plausibility_checks,
    )
