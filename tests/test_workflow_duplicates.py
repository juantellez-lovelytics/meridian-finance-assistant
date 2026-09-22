"""Tests for finance_assistant.workflows.duplicates.duplicate_payment_check (Q8).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/*.csv files. Expected candidate counts are hand-computed, never
derived by calling the function under test.
"""

from finance_assistant import config
from finance_assistant.data.loaders import load_gl_transactions, load_vendors
from finance_assistant.evidence.models import AnswerStatus
from finance_assistant.workflows.duplicates import duplicate_payment_check


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
        "vendor_id": "V1",
        "doc_ref": "D1",
        "approval_ref": "",
        "memo": "consulting fee",
    }
    row.update(overrides)
    return row


def _vendor_row(**overrides):
    row = {"vendor_id": "V1", "vendor_name": "Acme Inc", "category": "Services", "country": "US"}
    row.update(overrides)
    return row


def _load(write_csv, gl_rows, vendor_rows):
    gl_path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], gl_rows)
    vendors_path = write_csv(config.VENDORS_SCHEMA["filename"], config.VENDORS_SCHEMA["required_columns"], vendor_rows)
    return load_gl_transactions(gl_path), load_vendors(vendors_path)


def test_high_confidence_pair_detected(write_csv):
    gl, vendors = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", doc_ref="D1"),
            _gl_row(txn_id="T2", doc_ref="D2"),
        ],
        [_vendor_row()],
    )

    bundle = duplicate_payment_check(gl, vendors)

    assert bundle.status == AnswerStatus.ANSWER
    assert len(bundle.result["candidates_by_confidence"]["HIGH"]) == 1


def test_reversed_leg_is_surfaced_not_dropped(write_csv):
    gl, vendors = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", doc_ref="D1", amount="1000.00"),
            _gl_row(txn_id="T2", doc_ref="D2", amount="1000.00"),
            _gl_row(txn_id="T3", doc_ref="D3", amount="-1000.00", memo="reversal of D1"),
        ],
        [_vendor_row()],
    )

    bundle = duplicate_payment_check(gl, vendors)

    assert bundle.result["reversed_candidate_count"] == 1
    high = bundle.result["candidates_by_confidence"]["HIGH"]
    assert any(c["is_reversed"] for c in high)


def test_low_confidence_candidate_via_alias_cluster(write_csv):
    gl, vendors = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", vendor_id="V1", amount="750.00", memo="freight"),
            _gl_row(txn_id="T2", vendor_id="V2", amount="750.00", memo="logistics"),
        ],
        [
            _vendor_row(vendor_id="V1", vendor_name="Kestrel Freight Systems"),
            _vendor_row(vendor_id="V2", vendor_name="Kestrel Freight Systems, Inc."),
        ],
    )

    bundle = duplicate_payment_check(gl, vendors)

    assert len(bundle.result["candidates_by_confidence"]["LOW"]) == 1


def test_missing_evidence_always_present_even_with_zero_candidates(write_csv):
    gl, vendors = _load(write_csv, [_gl_row(txn_id="T1")], [_vendor_row()])

    bundle = duplicate_payment_check(gl, vendors)

    assert bundle.result["candidates_by_confidence"] == {"HIGH": [], "MEDIUM": [], "LOW": []}
    assert len(bundle.missing_evidence) == 2


def test_tool_calls_are_recorded(write_csv):
    gl, vendors = _load(
        write_csv,
        [
            _gl_row(txn_id="T1", doc_ref="D1"),
            _gl_row(txn_id="T2", doc_ref="D2"),
        ],
        [_vendor_row()],
    )

    bundle = duplicate_payment_check(gl, vendors, date_start="2024-04-01", date_end="2024-04-30")

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {"query_ledger", "detect_alias_clusters", "detect_duplicate_candidates"}
    assert "query_ledger" in tool_names
    assert "detect_duplicate_candidates" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    vendors = load_vendors()

    bundle = duplicate_payment_check(gl, vendors)

    assert bundle.status == AnswerStatus.ANSWER
    assert len(bundle.missing_evidence) == 2
