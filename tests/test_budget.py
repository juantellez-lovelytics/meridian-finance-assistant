"""Tests for finance_assistant.tools.budget.query_budget (R5).

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loader, never against the real data/*.csv
files. Expected sums and ratios are always hand-computed, never derived
by calling the function under test.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_budget
from finance_assistant.tools.budget import query_budget


def _budget_row(**overrides):
    row = {
        "entity": "E1",
        "cost_centre": "CC-A",
        "account_code": "AC1",
        "period_month": "2024-07",
        "budget_amount": "1000.00",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def _load_budget(write_csv, rows):
    path = write_csv(config.BUDGET_SCHEMA["filename"], config.BUDGET_SCHEMA["required_columns"], rows)
    return load_budget(path)


# ---------------------------------------------------------------------------
# Duplicate dimensional keys: detected generically, never drop_duplicates.
# ---------------------------------------------------------------------------


def test_duplicate_keys_detected_with_count_and_amount(write_csv):
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", budget_amount="600"),
            _budget_row(cost_centre="CC-A", budget_amount="400"),
            _budget_row(cost_centre="CC-B", budget_amount="2000"),
        ],
    )

    result = query_budget(budget, period_start="2024-07", period_end="2024-07")

    assert len(result.duplicate_keys) == 1
    dup = result.duplicate_keys[0]
    assert dup.cost_centre == "CC-A"
    assert dup.row_count == 2
    assert dup.total_budget_amount == pytest.approx(1000.0)


def test_raw_rows_never_dropped(write_csv):
    rows = [
        _budget_row(cost_centre="CC-A", budget_amount="600"),
        _budget_row(cost_centre="CC-A", budget_amount="400"),
    ]
    budget = _load_budget(write_csv, rows)

    result = query_budget(budget, period_start="2024-07", period_end="2024-07")

    assert len(result.rows) == 2


def test_aggregated_rows_apply_additive_rule(write_csv):
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", budget_amount="600"),
            _budget_row(cost_centre="CC-A", budget_amount="400"),
            _budget_row(cost_centre="CC-B", budget_amount="2000"),
        ],
    )

    result = query_budget(budget, period_start="2024-07", period_end="2024-07")

    assert result.aggregation_rule == "additive"
    by_cc = result.aggregated_rows.set_index("cost_centre")
    assert by_cc.loc["CC-A", "budget_amount"] == pytest.approx(1000.0)
    assert by_cc.loc["CC-A", "row_count"] == 2
    assert by_cc.loc["CC-B", "budget_amount"] == pytest.approx(2000.0)
    assert by_cc.loc["CC-B", "row_count"] == 1


# ---------------------------------------------------------------------------
# Filters: period range and dimensions.
# ---------------------------------------------------------------------------


def test_period_range_is_inclusive(write_csv):
    budget = _load_budget(
        write_csv,
        [
            _budget_row(period_month="2024-06", budget_amount="100"),
            _budget_row(period_month="2024-07", budget_amount="200"),
            _budget_row(period_month="2024-08", budget_amount="300"),
        ],
    )

    result = query_budget(budget, period_start="2024-06", period_end="2024-07")

    assert set(result.rows["period_month"]) == {"2024-06", "2024-07"}


def test_dimension_filters(write_csv):
    rows = [
        _budget_row(entity="E1", cost_centre="CC-A", account_code="AC1"),
        _budget_row(entity="E2", cost_centre="CC-B", account_code="AC2"),
    ]
    budget = _load_budget(write_csv, rows)

    result = query_budget(budget, period_start="2024-07", period_end="2024-07", cost_centres=["CC-B"])

    assert set(result.rows["cost_centre"]) == {"CC-B"}


def test_period_start_after_end_raises(write_csv):
    budget = _load_budget(write_csv, [_budget_row()])

    with pytest.raises(ValueError):
        query_budget(budget, period_start="2024-08", period_end="2024-07")


def test_non_usd_currency_raises(write_csv):
    budget = _load_budget(write_csv, [_budget_row(currency="EUR")])

    with pytest.raises(ValueError):
        query_budget(budget, period_start="2024-07", period_end="2024-07")


# ---------------------------------------------------------------------------
# R5 plausibility check: additive vs single-row hypothesis vs peer median.
# ---------------------------------------------------------------------------


def test_plausibility_check_prefers_hypothesis_closer_to_peer_median(write_csv):
    # CC-A is the affected (duplicated-key) centre: rows of 500 then 1000,
    # in that CSV order -> additive budget 1500, single-row (first) budget 500.
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="500"),
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="1000"),
            _budget_row(cost_centre="CC-B", account_code="AC1", budget_amount="2000"),
            _budget_row(cost_centre="CC-C", account_code="AC1", budget_amount="3000"),
        ],
    )
    actual_usd_by_cost_centre = {
        "CC-A": 1200.0,  # additive ratio 1200/1500=0.80, single ratio 1200/500=2.40
        "CC-B": 1800.0,  # ratio 0.90
        "CC-C": 2850.0,  # ratio 0.95
    }

    result = query_budget(
        budget,
        period_start="2024-07",
        period_end="2024-07",
        actual_usd_by_cost_centre=actual_usd_by_cost_centre,
    )

    assert len(result.plausibility_checks) == 1
    check = result.plausibility_checks[0]
    assert check.affected_cost_centre == "CC-A"
    assert check.additive_ratio == pytest.approx(0.8)
    assert check.single_row_ratio == pytest.approx(2.4)
    assert check.peer_median_ratio == pytest.approx(0.925)  # median(0.90, 0.95)
    assert check.preferred_hypothesis == "additive"
    assert check.is_ambiguous is False


def test_plausibility_check_flags_ambiguous_when_hypotheses_equally_plausible(write_csv):
    # additive budget = 875 + 125 = 1000 -> additive ratio 1400/1000 = 1.4
    # single-row (first row) budget = 875           -> single ratio 1400/875 = 1.6
    # peers both sit at ratio 1.5 -> median 1.5, and both hypotheses are
    # exactly 0.1 away from it: neither is more plausible.
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="875"),
            _budget_row(cost_centre="CC-A", account_code="AC1", budget_amount="125"),
            _budget_row(cost_centre="CC-B", account_code="AC1", budget_amount="1000"),
            _budget_row(cost_centre="CC-C", account_code="AC1", budget_amount="500"),
        ],
    )
    actual_usd_by_cost_centre = {
        "CC-A": 1400.0,
        "CC-B": 1500.0,  # ratio 1.5
        "CC-C": 750.0,  # ratio 1.5
    }

    result = query_budget(
        budget,
        period_start="2024-07",
        period_end="2024-07",
        actual_usd_by_cost_centre=actual_usd_by_cost_centre,
    )

    check = result.plausibility_checks[0]
    assert check.additive_ratio == pytest.approx(1.4)
    assert check.single_row_ratio == pytest.approx(1.6)
    assert check.peer_median_ratio == pytest.approx(1.5)
    assert check.is_ambiguous is True


def test_no_plausibility_checks_without_actuals(write_csv):
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", budget_amount="600"),
            _budget_row(cost_centre="CC-A", budget_amount="400"),
        ],
    )

    result = query_budget(budget, period_start="2024-07", period_end="2024-07")

    assert result.plausibility_checks == []


def test_no_plausibility_check_when_affected_centre_missing_from_actuals(write_csv):
    budget = _load_budget(
        write_csv,
        [
            _budget_row(cost_centre="CC-A", budget_amount="600"),
            _budget_row(cost_centre="CC-A", budget_amount="400"),
            _budget_row(cost_centre="CC-B", budget_amount="2000"),
        ],
    )

    result = query_budget(
        budget,
        period_start="2024-07",
        period_end="2024-07",
        actual_usd_by_cost_centre={"CC-B": 1800.0},  # CC-A (the affected centre) absent
    )

    assert result.plausibility_checks == []


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    budget = load_budget()

    result = query_budget(budget, period_start="2024-01", period_end="2024-12")

    assert result.aggregation_rule == "additive"
    assert len(result.rows) <= len(budget)

    for dup in result.duplicate_keys:
        assert dup.row_count >= 2
        matching_raw = result.rows[
            (result.rows["entity"] == dup.entity)
            & (result.rows["cost_centre"] == dup.cost_centre)
            & (result.rows["account_code"] == dup.account_code)
            & (result.rows["period_month"] == dup.period_month)
        ]
        assert len(matching_raw) == dup.row_count
        assert matching_raw["budget_amount"].sum() == pytest.approx(dup.total_budget_amount)

    assert len(result.aggregated_rows) == result.rows.drop_duplicates(
        ["entity", "cost_centre", "account_code", "period_month"]
    ).shape[0]
