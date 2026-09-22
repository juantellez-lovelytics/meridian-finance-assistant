"""Python-side wiring for the deterministic eval cases: which workflow each
case id calls, how to build its structured parameters from the loaded
`Dataset` (never a literal dataset value), and which precondition (if any)
decides which branch of questions.yaml applies.

Kept separate from questions.yaml because dynamic parameter derivation
("the two most recent years present in the ledger") requires code, not
YAML data. `id` is the join key between the two files.
"""

from dataclasses import dataclass
from typing import Callable

from finance_assistant.evidence.models import EvidenceBundle
from finance_assistant.workflows.consolidated import consolidated_spend
from finance_assistant.workflows.duplicates import duplicate_payment_check
from finance_assistant.workflows.headcount import headcount_cost_per_fte
from finance_assistant.workflows.opex import opex_by_cost_centre
from finance_assistant.workflows.policy import te_policy_check
from finance_assistant.workflows.travel import travel_comparison
from finance_assistant.workflows.variance import budget_variance
from finance_assistant.workflows.vendors import top_vendors

from evals.dataset import Dataset
from evals.preconditions import (
    PreconditionResult,
    alias_clusters_change_topn,
    coa_reclassification_between_years,
    fx_incomplete_or_budget_keys_ambiguous,
    missing_fx_rate_in_period,
    opex_q2_period_ambiguous,
)

TOP_N_VENDORS = 10


@dataclass(frozen=True)
class Case:
    id: str
    workflow: Callable[..., EvidenceBundle]
    build_params: Callable[[Dataset], dict]
    precondition: Callable[[Dataset], PreconditionResult] | None = None


def _q1_params(ds: Dataset) -> dict:
    return {"gl": ds.gl, "coa": ds.coa, "fx": ds.fx, "quarter": "Q2", "year": None}


def _q1_precondition(ds: Dataset) -> PreconditionResult:
    return opex_q2_period_ambiguous(ds.gl, ds.coa, ds.fx, "Q2")


def _q2_params(ds: Dataset) -> dict:
    year_current, year_prior = ds.two_most_recent_years()
    return {"gl": ds.gl, "coa": ds.coa, "fx": ds.fx, "year_current": year_current, "year_prior": year_prior}


def _q2_precondition(ds: Dataset) -> PreconditionResult:
    year_current, year_prior = ds.two_most_recent_years()
    return coa_reclassification_between_years(ds.coa, year_prior=year_prior, year_current=year_current)


def _q3_params(ds: Dataset) -> dict:
    return {"gl": ds.gl, "fx": ds.fx, "quarter": "Q3", "year": None}


def _q3_precondition(ds: Dataset) -> PreconditionResult:
    return missing_fx_rate_in_period(ds.gl, ds.fx, "Q3")


def _q4_params(ds: Dataset) -> dict:
    date_start, date_end = ds.full_date_range()
    return {"gl": ds.gl, "vendors": ds.vendors, "fx": ds.fx, "date_start": date_start, "date_end": date_end, "top_n": TOP_N_VENDORS}


def _q4_precondition(ds: Dataset) -> PreconditionResult:
    date_start, date_end = ds.full_date_range()
    return alias_clusters_change_topn(ds.gl, ds.vendors, ds.fx, date_start, date_end, top_n=TOP_N_VENDORS)


def _q5_params(ds: Dataset) -> dict:
    return {
        "gl": ds.gl,
        "coa": ds.coa,
        "fx": ds.fx,
        "budget": ds.budget,
        "quarter": "Q3",
        "year": ds.budget_year(),
        "documents_dir": ds.documents_dir,
    }


def _q5_precondition(ds: Dataset) -> PreconditionResult:
    return fx_incomplete_or_budget_keys_ambiguous(ds.gl, ds.coa, ds.fx, ds.budget, "Q3", ds.budget_year())


def _q6_params(ds: Dataset) -> dict:
    date_start, date_end = ds.full_date_range()
    return {"gl": ds.gl, "coa": ds.coa, "fx": ds.fx, "date_start": date_start, "date_end": date_end}


def _q7_params(ds: Dataset) -> dict:
    return {"documents_dir": ds.documents_dir}


def _q8_params(ds: Dataset) -> dict:
    return {"gl": ds.gl, "vendors": ds.vendors}


CASES: dict[str, Case] = {
    "q1_opex_by_cost_centre": Case("q1_opex_by_cost_centre", opex_by_cost_centre, _q1_params, _q1_precondition),
    "q2_travel_comparison": Case("q2_travel_comparison", travel_comparison, _q2_params, _q2_precondition),
    "q3_consolidated_spend": Case("q3_consolidated_spend", consolidated_spend, _q3_params, _q3_precondition),
    "q4_top_vendors": Case("q4_top_vendors", top_vendors, _q4_params, _q4_precondition),
    "q5_budget_variance": Case("q5_budget_variance", budget_variance, _q5_params, _q5_precondition),
    "q6_te_policy_check": Case("q6_te_policy_check", te_policy_check, _q6_params, None),
    "q7_headcount_cost_per_fte": Case("q7_headcount_cost_per_fte", headcount_cost_per_fte, _q7_params, None),
    "q8_duplicate_payment_check": Case("q8_duplicate_payment_check", duplicate_payment_check, _q8_params, None),
}
