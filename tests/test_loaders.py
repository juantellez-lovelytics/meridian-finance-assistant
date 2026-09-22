"""Loader tests.

Hard rule: no assertion here may depend on a concrete value or row count
from the real data/*.csv files, because the evaluator reruns this suite
against a second dataset with the same schema and different numbers.
Against real data/ we only assert structural invariants that hold for any
dataset sharing this schema. Concrete counts/values are covered separately
via synthetic tmp_path fixtures.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import (
    load_budget,
    load_chart_of_accounts,
    load_fx_rates,
    load_gl_transactions,
    load_vendors,
)
from finance_assistant.data.validation import InvalidColumnTypeError, MissingColumnsError

LOADERS = [
    (load_gl_transactions, config.GL_SCHEMA),
    (load_chart_of_accounts, config.COA_SCHEMA),
    (load_budget, config.BUDGET_SCHEMA),
    (load_fx_rates, config.FX_SCHEMA),
    (load_vendors, config.VENDORS_SCHEMA),
]


# ---------------------------------------------------------------------------
# Happy path against real data/ — structural invariants only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loader_fn,schema", LOADERS, ids=[s["filename"] for _, s in LOADERS])
def test_loader_happy_path_structural_invariants(loader_fn, schema):
    df = loader_fn()

    assert len(df) > 0
    assert set(schema["required_columns"]) <= set(df.columns)

    for col in schema["date_columns"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} not coerced to datetime"

    for col in schema["numeric_columns"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not coerced to numeric"

    if "account_code" in df.columns:
        assert df["account_code"].dtype == "string"


def test_gl_key_columns_have_no_nulls():
    df = load_gl_transactions()
    for col in ["txn_id", "account_code", "entity", "currency"]:
        assert df[col].notna().all(), f"{col} has unexpected nulls"


def test_gl_txn_id_is_unique():
    df = load_gl_transactions()
    assert df["txn_id"].is_unique


def test_coa_key_columns_have_no_nulls():
    df = load_chart_of_accounts()
    for col in ["account_code"]:
        assert df[col].notna().all(), f"{col} has unexpected nulls"


def test_budget_key_columns_have_no_nulls():
    df = load_budget()
    for col in ["entity", "cost_centre", "account_code", "currency"]:
        assert df[col].notna().all(), f"{col} has unexpected nulls"


def test_fx_key_columns_have_no_nulls():
    df = load_fx_rates()
    for col in ["period_month", "currency"]:
        assert df[col].notna().all(), f"{col} has unexpected nulls"


def test_vendors_key_columns_have_no_nulls():
    df = load_vendors()
    assert df["vendor_id"].notna().all()


# ---------------------------------------------------------------------------
# Missing required column -> MissingColumnsError. Synthetic data only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loader_fn,schema", LOADERS, ids=[s["filename"] for _, s in LOADERS])
def test_missing_required_column_raises(loader_fn, schema, write_csv):
    dropped = schema["required_columns"][0]
    header = [c for c in schema["required_columns"] if c != dropped]
    path = write_csv(schema["filename"], header, rows=[{c: "x" for c in header}])

    with pytest.raises(MissingColumnsError) as exc_info:
        loader_fn(path)

    assert schema["filename"] in str(exc_info.value)
    assert dropped in str(exc_info.value)


# ---------------------------------------------------------------------------
# Synthetic fixtures: concrete counts/values live only here, never against
# the real dataset.
# ---------------------------------------------------------------------------


def _valid_coa_row(**overrides):
    row = {
        "account_code": "1000",
        "account_name": "Test Account",
        "parent_code": "1000",
        "parent_name": "Test Parent",
        "statement_line": "Operating Expenses",
        "valid_from": "2024-01-01",
        "valid_to": "2024-12-31",
    }
    row.update(overrides)
    return row


def test_coa_sentinel_valid_to_parses(write_csv):
    row = _valid_coa_row(valid_to="9999-12-31")
    path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], [row])

    df = load_chart_of_accounts(path)

    assert df["valid_to"].iloc[0] == pd.Timestamp("9999-12-31")


def test_coa_invalid_date_raises(write_csv):
    row = _valid_coa_row(valid_from="not-a-date")
    path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], [row])

    with pytest.raises(InvalidColumnTypeError):
        load_chart_of_accounts(path)


def _valid_gl_row(**overrides):
    row = {
        "txn_id": "T0001",
        "posting_date": "2024-01-05",
        "accrual_date": "2024-01-01",
        "entity": "MI-US",
        "cost_centre": "OPS-NA",
        "account_code": "6320",
        "amount": "100.00",
        "currency": "USD",
        "vendor_id": "V1001",
        "doc_ref": "INV-0001",
        "approval_ref": "",
        "memo": "test row",
    }
    row.update(overrides)
    return row


def test_gl_loader_returns_exact_row_count_for_synthetic_fixture(write_csv):
    rows = [_valid_gl_row(txn_id=f"T{i:04d}") for i in range(3)]
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)

    df = load_gl_transactions(path)

    assert len(df) == 3


def test_gl_loader_coerces_blank_vendor_id_to_na(write_csv):
    rows = [_valid_gl_row(vendor_id="")]
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)

    df = load_gl_transactions(path)

    assert pd.isna(df["vendor_id"].iloc[0])


def test_gl_loader_rejects_non_numeric_amount(write_csv):
    rows = [_valid_gl_row(amount="not-a-number")]
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)

    with pytest.raises(InvalidColumnTypeError):
        load_gl_transactions(path)
