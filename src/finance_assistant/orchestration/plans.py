"""The deterministic intent -> workflow registry (the "Plan Registry").

`REGISTRY`/`IntentSpec`/`LOADERS` are the single source of truth for how an
intent maps to a workflow and which DataFrames/documents_dir it needs
injected -- this used to live duplicated as private names inside
`cli.py`; `cli.py` now imports it from here instead.

Also owns dataset-derived parameter defaulting: several real questions
name no literal year/date at all ("the most recent year", no period
stated), yet the underlying workflows require those as non-defaulted
kwargs. The Question Interpreter is forbidden from seeing dataset years,
so it can never supply them -- this module derives them from whatever is
actually loaded, the same mechanism `evals/dataset.py` already used for
the eval harness (which now delegates here instead of duplicating it).
"""

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from finance_assistant import config
from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.data.loaders import (
    load_budget,
    load_chart_of_accounts,
    load_fx_rates,
    load_gl_transactions,
    load_vendors,
)
from finance_assistant.evidence.models import EvidenceBundle
from finance_assistant.orchestration.intents import Intent, IntentRequest
from finance_assistant.workflows.consolidated import consolidated_spend
from finance_assistant.workflows.duplicates import duplicate_payment_check
from finance_assistant.workflows.headcount import headcount_cost_per_fte
from finance_assistant.workflows.opex import opex_by_cost_centre
from finance_assistant.workflows.policy import te_policy_check
from finance_assistant.workflows.travel import travel_comparison
from finance_assistant.workflows.variance import budget_variance
from finance_assistant.workflows.vendors import top_vendors


class PlanResolutionError(Exception):
    """A user-facing plan/parameter error -- caught by cli.py alongside its
    own CliError, and by orchestrator.py to build an ERROR-status bundle."""


@dataclass(frozen=True)
class IntentSpec:
    workflow: Callable[..., EvidenceBundle]
    dataframe_args: tuple[str, ...]
    documents_dir_arg: str | None = None


LOADERS: dict[str, tuple[Callable[..., "pd.DataFrame"], str]] = {
    "gl": (load_gl_transactions, config.GL_TRANSACTIONS_FILE),
    "coa": (load_chart_of_accounts, config.CHART_OF_ACCOUNTS_FILE),
    "fx": (load_fx_rates, config.FX_RATES_FILE),
    "budget": (load_budget, config.BUDGET_FILE),
    "vendors": (load_vendors, config.VENDORS_FILE),
}

REGISTRY: dict[Intent, IntentSpec] = {
    Intent.OPEX_BY_COST_CENTRE: IntentSpec(opex_by_cost_centre, ("gl", "coa", "fx")),
    Intent.TRAVEL_COMPARISON: IntentSpec(travel_comparison, ("gl", "coa", "fx")),
    Intent.CONSOLIDATED_SPEND: IntentSpec(consolidated_spend, ("gl", "fx")),
    Intent.TOP_VENDORS: IntentSpec(top_vendors, ("gl", "vendors", "fx")),
    Intent.BUDGET_VARIANCE: IntentSpec(budget_variance, ("gl", "coa", "fx", "budget"), documents_dir_arg="documents_dir"),
    Intent.TE_POLICY_CHECK: IntentSpec(te_policy_check, ("gl", "coa", "fx")),
    Intent.HEADCOUNT_COST_PER_FTE: IntentSpec(headcount_cost_per_fte, (), documents_dir_arg="documents_dir"),
    Intent.DUPLICATE_PAYMENT_CHECK: IntentSpec(duplicate_payment_check, ("gl", "vendors")),
}


def load_dataframes(spec: IntentSpec, data_dir: Path) -> dict[str, "pd.DataFrame"]:
    return {arg: LOADERS[arg][0](data_dir / LOADERS[arg][1]) for arg in spec.dataframe_args}


# --- dataset-derived defaults: only for params the interpreter can never
# legitimately supply (it never sees dataset years/dates). Never override
# an LLM-stated value -- callers merge as {**default_params(...), **llm_params}.
# Never raise -- an unresolvable default silently yields {}, so the single
# "missing required parameter" error from assemble_kwargs is what surfaces
# the problem, instead of two independent error paths. -----------------


def default_two_most_recent_years(gl: "pd.DataFrame", date_field: str = DEFAULT_FINANCIAL_DATE_FIELD) -> dict:
    years = sorted({int(y) for y in gl[date_field].dropna().dt.year.unique()}, reverse=True)
    if len(years) < 2:
        return {}
    return {"year_current": years[0], "year_prior": years[1]}


def default_budget_year(budget: "pd.DataFrame") -> dict:
    years = sorted({str(pm)[:4] for pm in budget["period_month"].dropna().unique()})
    if len(years) != 1:
        return {}
    return {"year": int(years[0])}


def default_full_date_range(gl: "pd.DataFrame", date_field: str = DEFAULT_FINANCIAL_DATE_FIELD) -> dict:
    series = gl[date_field].dropna()
    if series.empty:
        return {}
    return {"date_start": series.min().strftime("%Y-%m-%d"), "date_end": series.max().strftime("%Y-%m-%d")}


_DATASET_DEFAULTERS: dict[Intent, Callable[[dict[str, "pd.DataFrame"]], dict]] = {
    Intent.TRAVEL_COMPARISON: lambda dfs: default_two_most_recent_years(dfs["gl"]),
    Intent.TOP_VENDORS: lambda dfs: default_full_date_range(dfs["gl"]),
    Intent.TE_POLICY_CHECK: lambda dfs: default_full_date_range(dfs["gl"]),
    Intent.BUDGET_VARIANCE: lambda dfs: default_budget_year(dfs["budget"]),
}


def default_params(intent: Intent, dataframes: dict[str, "pd.DataFrame"]) -> dict:
    defaulter = _DATASET_DEFAULTERS.get(intent)
    return defaulter(dataframes) if defaulter else {}


def extract_llm_params(request: IntentRequest, spec: IntentSpec) -> dict:
    """Flattens `IntentRequest` to only the fields legal for the resolved
    intent's workflow signature, dropping fields irrelevant to this intent
    (e.g. a stray `top_n` on an opex request) instead of erroring -- the
    flat schema is expected to carry noise for other intents; that isn't
    malformed input worth failing on."""
    candidate = request.model_dump(exclude={"intent", "confidence"}, exclude_none=True)
    valid_names = set(inspect.signature(spec.workflow).parameters) - set(spec.dataframe_args)
    if spec.documents_dir_arg:
        valid_names.discard(spec.documents_dir_arg)
    return {k: v for k, v in candidate.items() if k in valid_names}


def assemble_kwargs(spec: IntentSpec, params: dict, dataframes: dict[str, "pd.DataFrame"], documents_dir: Path) -> dict:
    """Same validation the CLI's former `_build_kwargs` already had
    (collision check, unknown/missing param checks against
    `inspect.signature`) -- messages UNCHANGED from before this move, since
    `tests/test_cli.py` asserts on exact substrings."""
    collisions = set(params) & set(spec.dataframe_args)
    if collisions:
        raise PlanResolutionError(
            f"params must not include dataframe argument(s) {sorted(collisions)} — "
            "the CLI injects gl/coa/fx/budget/vendors from --data-dir automatically"
        )

    kwargs: dict[str, object] = dict(dataframes)
    if spec.documents_dir_arg and spec.documents_dir_arg not in params:
        kwargs[spec.documents_dir_arg] = documents_dir
    kwargs.update(params)

    signature = inspect.signature(spec.workflow)
    unknown = set(kwargs) - set(signature.parameters)
    if unknown:
        raise PlanResolutionError(f"unknown parameter(s) for {spec.workflow.__name__}: {sorted(unknown)}")

    required = {name for name, p in signature.parameters.items() if p.default is inspect.Parameter.empty}
    missing = required - set(kwargs)
    if missing:
        raise PlanResolutionError(f"missing required parameter(s) for {spec.workflow.__name__}: {sorted(missing)}")

    return kwargs


def build_workflow_kwargs(spec: IntentSpec, params: dict, data_dir: Path) -> dict:
    """Convenience wrapper preserving the JSON-mode call shape (spec,
    params, data_dir) -> kwargs -- JSON mode needs no dataset-derived
    defaulting, since a JSON question always states params explicitly."""
    dataframes = load_dataframes(spec, data_dir)
    documents_dir = data_dir / "documents"
    return assemble_kwargs(spec, params, dataframes, documents_dir)
