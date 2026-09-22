"""Tests for finance_assistant.orchestration.plans (the deterministic
intent -> workflow registry, and dataset-derived parameter defaulting).

All dataframes are built via the write_csv fixture under tmp_path -- never
the real data/*.csv files.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_budget, load_gl_transactions
from finance_assistant.orchestration.intents import Intent, IntentRequest
from finance_assistant.orchestration.plans import (
    REGISTRY,
    PlanResolutionError,
    assemble_kwargs,
    default_budget_year,
    default_full_date_range,
    default_params,
    default_two_most_recent_years,
    extract_llm_params,
)


def _gl_row(**overrides):
    row = {
        "txn_id": "T1",
        "posting_date": "2024-04-05",
        "accrual_date": "2024-04-05",
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "amount": "1000.00",
        "currency": "USD",
        "vendor_id": "",
        "doc_ref": "D1",
        "approval_ref": "",
        "memo": "",
    }
    row.update(overrides)
    return row


def _budget_row(**overrides):
    row = {
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "period_month": "2024-04",
        "budget_amount": "500.00",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def test_registry_covers_all_eight_non_unknown_intents():
    expected = {i for i in Intent if i is not Intent.UNKNOWN}
    assert set(REGISTRY) == expected


def test_default_two_most_recent_years_from_two_distinct_years(write_csv):
    path = write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [_gl_row(accrual_date="2023-05-01"), _gl_row(accrual_date="2024-05-01")],
    )
    gl = load_gl_transactions(path)
    assert default_two_most_recent_years(gl) == {"year_current": 2024, "year_prior": 2023}


def test_default_two_most_recent_years_returns_empty_when_only_one_year(write_csv):
    path = write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [_gl_row(accrual_date="2024-05-01")],
    )
    gl = load_gl_transactions(path)
    assert default_two_most_recent_years(gl) == {}


def test_default_budget_year_from_single_year(write_csv):
    path = write_csv(
        config.BUDGET_SCHEMA["filename"],
        config.BUDGET_SCHEMA["required_columns"],
        [_budget_row(period_month="2024-01"), _budget_row(period_month="2024-06")],
    )
    budget = load_budget(path)
    assert default_budget_year(budget) == {"year": 2024}


def test_default_budget_year_returns_empty_when_multiple_years(write_csv):
    path = write_csv(
        config.BUDGET_SCHEMA["filename"],
        config.BUDGET_SCHEMA["required_columns"],
        [_budget_row(period_month="2023-12"), _budget_row(period_month="2024-01")],
    )
    budget = load_budget(path)
    assert default_budget_year(budget) == {}


def test_default_full_date_range(write_csv):
    path = write_csv(
        config.GL_SCHEMA["filename"],
        config.GL_SCHEMA["required_columns"],
        [_gl_row(accrual_date="2024-01-15"), _gl_row(accrual_date="2024-03-20")],
    )
    gl = load_gl_transactions(path)
    assert default_full_date_range(gl) == {"date_start": "2024-01-15", "date_end": "2024-03-20"}


def test_default_full_date_range_returns_empty_when_no_rows(write_csv):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], [])
    gl = load_gl_transactions(path)
    assert default_full_date_range(gl) == {}


def test_default_params_returns_empty_for_intent_with_no_defaulter(write_csv):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], [_gl_row()])
    gl = load_gl_transactions(path)
    assert default_params(Intent.OPEX_BY_COST_CENTRE, {"gl": gl}) == {}


def test_extract_llm_params_keeps_only_fields_relevant_to_resolved_intent():
    spec = REGISTRY[Intent.TRAVEL_COMPARISON]
    request = IntentRequest(
        intent=Intent.TRAVEL_COMPARISON,
        confidence=0.9,
        quarter="Q2",
        year=2024,
        year_current=2024,
        year_prior=2023,
        top_n=5,
    )
    assert extract_llm_params(request, spec) == {"year_current": 2024, "year_prior": 2023}


def test_extract_llm_params_drops_all_fields_when_none_apply():
    spec = REGISTRY[Intent.HEADCOUNT_COST_PER_FTE]
    request = IntentRequest(intent=Intent.HEADCOUNT_COST_PER_FTE, confidence=0.9, quarter="Q2", year=2024)
    assert extract_llm_params(request, spec) == {}


def test_assemble_kwargs_rejects_dataframe_key_collision():
    spec = REGISTRY[Intent.OPEX_BY_COST_CENTRE]
    dataframes = {"gl": object(), "coa": object(), "fx": object()}
    with pytest.raises(PlanResolutionError, match="dataframe"):
        assemble_kwargs(spec, {"gl": object(), "quarter": "Q2"}, dataframes, documents_dir=None)


def test_assemble_kwargs_rejects_unknown_param():
    spec = REGISTRY[Intent.OPEX_BY_COST_CENTRE]
    dataframes = {"gl": object(), "coa": object(), "fx": object()}
    with pytest.raises(PlanResolutionError, match="unknown parameter"):
        assemble_kwargs(spec, {"quarter": "Q2", "not_a_real_param": 1}, dataframes, documents_dir=None)


def test_assemble_kwargs_rejects_missing_required_param():
    spec = REGISTRY[Intent.OPEX_BY_COST_CENTRE]
    dataframes = {"gl": object(), "coa": object(), "fx": object()}
    with pytest.raises(PlanResolutionError, match="missing required parameter"):
        assemble_kwargs(spec, {}, dataframes, documents_dir=None)


def test_assemble_kwargs_injects_documents_dir_when_not_supplied(tmp_path):
    spec = REGISTRY[Intent.HEADCOUNT_COST_PER_FTE]
    documents_dir = tmp_path / "documents"
    kwargs = assemble_kwargs(spec, {}, {}, documents_dir=documents_dir)
    assert kwargs["documents_dir"] == documents_dir
