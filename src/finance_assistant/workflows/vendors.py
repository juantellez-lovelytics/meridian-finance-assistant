"""Q4 — top vendors, ranked two ways.

PARTIAL: ranking by vendor_id (authoritative, R6) and ranking by candidate
alias cluster (detected, never applied — a vendor never gets merged into
another's vendor_id) are both computed and returned. If the two top-N
compositions differ, the "top vendors" answer genuinely depends on a
non-authoritative grouping decision and must say so rather than presenting
one ranking as if it were definitive.
"""

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.evidence.gate import apply_gate
from finance_assistant.evidence.models import AnswerStatus, Coverage, EvidenceBundle
from finance_assistant.orchestration.intents import Intent
from finance_assistant.tools.fx import aggregate_usd, aggregate_usd_by, convert_to_usd
from finance_assistant.tools.ledger import query_ledger
from finance_assistant.tools.vendors import detect_alias_clusters, normalize_vendor_name, vendor_lookup
from finance_assistant.workflows._shared import ToolTrace


def top_vendors(
    gl: pd.DataFrame,
    vendors: pd.DataFrame,
    fx: pd.DataFrame,
    date_start: str,
    date_end: str,
    top_n: int = 10,
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> EvidenceBundle:
    tt = ToolTrace()

    ledger = tt.call(query_ledger, gl, date_start, date_end, date_field=date_field)
    lookup = tt.call(vendor_lookup, ledger.rows, vendors)
    fx_result = tt.call(convert_to_usd, lookup.rows, fx, date_field=date_field)

    has_vendor_id = fx_result.rows["vendor_id"].notna()
    vendor_id_rows = fx_result.rows.loc[has_vendor_id]
    vendor_id_fx = fx_result.__class__(rows=vendor_id_rows, coverage=fx_result.coverage, missing=fx_result.missing)
    ranking_by_vendor_id = tt.call(aggregate_usd_by, vendor_id_fx, by=["vendor_id"])

    alias_clusters = tt.call(detect_alias_clusters, vendors)
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
    ranking_by_cluster = tt.call(aggregate_usd_by, cluster_fx, by=["_cluster_key"])

    def top(ranking: dict) -> list[tuple]:
        return sorted(ranking.items(), key=lambda kv: kv[1].converted_amount_usd, reverse=True)[:top_n]

    top_vendor_id = top(ranking_by_vendor_id)
    top_cluster = top(ranking_by_cluster)

    top_vendor_id_set = {k[0] for k, _ in top_vendor_id}
    cluster_key_to_vendor_ids: dict[str, set[str]] = {}
    for vid, key in zip(working["vendor_id"], working["_cluster_key"]):
        cluster_key_to_vendor_ids.setdefault(key, set()).add(vid)
    top_cluster_vendor_ids: set[str] = set()
    for (key,), _ in top_cluster:
        top_cluster_vendor_ids |= cluster_key_to_vendor_ids.get(key, set())

    composition_changes = top_vendor_id_set != top_cluster_vendor_ids

    warnings: list[str] = [alias_clusters.limitation]

    def _frontier_warning(ranked: list[tuple], basis_name: str) -> None:
        if len(ranked) <= top_n:
            return
        boundary = ranked[top_n - 1 : top_n + 1]
        for key, amount in boundary:
            if amount.coverage.convertible_rows < amount.coverage.selected_rows:
                warnings.append(
                    f"{basis_name} ranking: {key[0]!r} sits at the top-{top_n} boundary with incomplete FX "
                    f"coverage ({amount.coverage.convertible_rows}/{amount.coverage.selected_rows} rows) — its true "
                    "position relative to its neighbor is uncertain"
                )

    all_ranked_vendor_id = sorted(ranking_by_vendor_id.items(), key=lambda kv: kv[1].converted_amount_usd, reverse=True)
    all_ranked_cluster = sorted(ranking_by_cluster.items(), key=lambda kv: kv[1].converted_amount_usd, reverse=True)
    _frontier_warning(all_ranked_vendor_id, "vendor_id")
    _frontier_warning(all_ranked_cluster, "alias-cluster")

    no_vendor_id_rows = fx_result.rows.loc[~has_vendor_id]
    no_vendor_id_fx = fx_result.__class__(rows=no_vendor_id_rows, coverage=fx_result.coverage, missing=fx_result.missing)
    vendor_less = tt.call(aggregate_usd, no_vendor_id_fx) if len(no_vendor_id_rows) else None

    coverage = Coverage.from_fx_coverage(fx_result.coverage)

    result = {
        "period": {"start": str(date_start), "end": str(date_end)},
        "top_n": top_n,
        "ranking_by_vendor_id": [{"vendor_id": k[0], "amount_usd": v.converted_amount_usd} for k, v in top_vendor_id],
        "ranking_by_alias_cluster": [{"cluster_key": k[0], "amount_usd": v.converted_amount_usd} for k, v in top_cluster],
        "composition_changes": composition_changes,
        "vendor_less_spend_usd": vendor_less.converted_amount_usd if vendor_less else 0.0,
        "unmatched_vendor_id_rows": lookup.unmatched_vendor_id_rows,
    }

    assumptions = [
        "ranking by vendor_id is authoritative (R6); the alias-cluster ranking is a candidate view only "
        "— no canonical_vendor_id exists in this dataset",
    ]

    gate_result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        grouping_would_change_result=composition_changes,
        fx_coverage=coverage,
        missing_fx_affects_requested_total=not fx_result.coverage.is_complete,
    )

    return EvidenceBundle(
        status=gate_result.final_status,
        intent=Intent.TOP_VENDORS,
        result=result if gate_result.final_status not in (AnswerStatus.REFUSED, AnswerStatus.NEEDS_CLARIFICATION) else None,
        assumptions=assumptions,
        warnings=warnings + gate_result.warnings_added,
        coverage=coverage,
        refusal_reason=None,
        clarification_options=gate_result.clarification_options,
        tool_calls=tt.calls,
    )
