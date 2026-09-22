"""Tests for finance_assistant.workflows.vendors.top_vendors (Q4).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected rankings are hand-computed, never derived
by calling the function under test.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_fx_rates, load_gl_transactions, load_vendors
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.workflows.vendors import top_vendors


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


def _vendor_row(**overrides):
    row = {"vendor_id": "V1", "vendor_name": "Acme Inc", "category": "Services", "country": "US"}
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-04", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, vendor_rows, fx_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    vendors_path = write_csv(config.VENDORS_SCHEMA["filename"], config.VENDORS_SCHEMA["required_columns"], vendor_rows)
    fx_path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], fx_rows)
    return load_gl_transactions(gl_path), load_vendors(vendors_path), load_fx_rates(fx_path)


def test_aliasable_vendors_change_top_n_composition(write_csv):
    # V1/V2 are the same vendor under two name spellings, individually below
    # V3 but combined above it -> cluster ranking's top-1 differs from
    # vendor_id ranking's top-1.
    gl, vendors, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", amount="600.00"),
            _gl_row(txn_id="T2", vendor_id="V2", amount="600.00"),
            _gl_row(txn_id="T3", vendor_id="V3", amount="1000.00"),
        ],
        [
            _vendor_row(vendor_id="V1", vendor_name="Acme Inc"),
            _vendor_row(vendor_id="V2", vendor_name="Acme, Inc."),
            _vendor_row(vendor_id="V3", vendor_name="Zenith Corp"),
        ],
        [_fx_row()],
    )

    bundle = top_vendors(gl, vendors, fx, "2024-04-01", "2024-04-30", top_n=1)

    assert bundle.status == AnswerStatus.PARTIAL
    assert bundle.result["composition_changes"] is True
    assert bundle.result["ranking_by_vendor_id"][0]["vendor_id"] == "V3"
    assert bundle.result["ranking_by_alias_cluster"][0]["cluster_key"] == "acme"


def test_no_aliasing_leaves_composition_unchanged(write_csv):
    gl, vendors, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", amount="1000.00"),
            _gl_row(txn_id="T2", vendor_id="V2", amount="500.00"),
        ],
        [
            _vendor_row(vendor_id="V1", vendor_name="Acme Inc"),
            _vendor_row(vendor_id="V2", vendor_name="Zenith Corp"),
        ],
        [_fx_row()],
    )

    bundle = top_vendors(gl, vendors, fx, "2024-04-01", "2024-04-30", top_n=2)

    assert bundle.result["composition_changes"] is False


def test_frontier_vendor_with_missing_fx_gets_warning(write_csv):
    gl, vendors, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", currency="USD", amount="1000.00"),
            _gl_row(txn_id="T2", vendor_id="V2", currency="EUR", amount="900.00"),  # no EUR rate -> frontier at top_n=1
        ],
        [
            _vendor_row(vendor_id="V1", vendor_name="Acme Inc"),
            _vendor_row(vendor_id="V2", vendor_name="Zenith Corp"),
        ],
        [_fx_row(currency="USD")],
    )

    bundle = top_vendors(gl, vendors, fx, "2024-04-01", "2024-04-30", top_n=1)

    assert any("V2" in w for w in bundle.warnings)


def test_rows_without_vendor_id_excluded_from_rankings(write_csv):
    gl, vendors, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", amount="1000.00"),
            _gl_row(txn_id="T2", vendor_id="", amount="5000.00"),  # payroll-like, no vendor
        ],
        [_vendor_row(vendor_id="V1", vendor_name="Acme Inc")],
        [_fx_row()],
    )

    bundle = top_vendors(gl, vendors, fx, "2024-04-01", "2024-04-30", top_n=5)

    assert bundle.result["vendor_less_spend_usd"] == pytest.approx(5000.0)
    assert len(bundle.result["ranking_by_vendor_id"]) == 1


def test_tool_calls_are_recorded(write_csv):
    gl, vendors, fx = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", amount="1000.00"),
            _gl_row(txn_id="T2", vendor_id="V2", amount="500.00"),
        ],
        [
            _vendor_row(vendor_id="V1", vendor_name="Acme Inc"),
            _vendor_row(vendor_id="V2", vendor_name="Zenith Corp"),
        ],
        [_fx_row()],
    )

    bundle = top_vendors(gl, vendors, fx, "2024-04-01", "2024-04-30", top_n=2)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {
        "query_ledger",
        "vendor_lookup",
        "convert_to_usd",
        "aggregate_usd_by",
        "detect_alias_clusters",
        "aggregate_usd",
    }
    assert "query_ledger" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    vendors = load_vendors()
    fx = load_fx_rates()

    bundle = top_vendors(gl, vendors, fx, "2024-01-01", "2024-12-31")

    assert bundle.status in (AnswerStatus.ANSWER, AnswerStatus.PARTIAL)
    assert bundle.result is not None
    assert len(bundle.result["ranking_by_vendor_id"]) <= 10
    assert len(bundle.result["ranking_by_alias_cluster"]) <= 10
