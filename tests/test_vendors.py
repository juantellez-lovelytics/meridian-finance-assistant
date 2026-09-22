"""Tests for finance_assistant.tools.vendors (R6): vendor_lookup (left join)
and detect_alias_clusters (deterministic candidate detection, never applied).

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loaders, never against the real data/*.csv
files.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_gl_transactions, load_vendors
from finance_assistant.tools.vendors import (
    AliasCluster,
    detect_alias_clusters,
    normalize_vendor_name,
    vendor_lookup,
)


def _gl_row(**overrides):
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


def _vendor_row(**overrides):
    row = {"vendor_id": "V1001", "vendor_name": "Acme Inc.", "category": "Software", "country": "US"}
    row.update(overrides)
    return row


def _load_gl(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


def _load_vendors(write_csv, rows):
    path = write_csv(config.VENDORS_SCHEMA["filename"], config.VENDORS_SCHEMA["required_columns"], rows)
    return load_vendors(path)


# ---------------------------------------------------------------------------
# normalize_vendor_name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Acme Inc.", "acme"),
        ("ACME", "acme"),
        ("Northwind Traders LLC", "northwind traders"),
        ("Northwind Traders", "northwind traders"),
        ("Beta, B.V.", "beta"),
        ("Sorensen & Hale", "sorensen hale"),
    ],
)
def test_normalize_vendor_name(raw, expected):
    assert normalize_vendor_name(raw) == expected


# ---------------------------------------------------------------------------
# vendor_lookup: left join, never drops vendor-less or unmatched rows.
# ---------------------------------------------------------------------------


def test_matched_vendor_id_is_enriched(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row(vendor_id="V1001", vendor_name="Acme Inc.")])
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_MATCHED", vendor_id="V1001")])

    result = vendor_lookup(gl, vendors)

    row = result.rows.set_index("txn_id").loc["T_MATCHED"]
    assert row["vendor_name"] == "Acme Inc."
    assert result.total_rows == 1
    assert result.matched_rows == 1
    assert result.no_vendor_id_rows == 0
    assert result.unmatched_vendor_id_rows == 0


def test_missing_vendor_id_is_preserved_as_legitimate_spend(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row()])
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_NO_VENDOR", vendor_id="")])

    result = vendor_lookup(gl, vendors)

    assert result.total_rows == 1
    assert result.no_vendor_id_rows == 1
    assert result.matched_rows == 0
    assert result.unmatched_vendor_id_rows == 0
    # Row survives the join -- a left join, never an inner join.
    assert "T_NO_VENDOR" in set(result.rows["txn_id"])


def test_unmatched_vendor_id_is_kept_and_flagged(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row(vendor_id="V1001")])
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_UNKNOWN_VENDOR", vendor_id="V9999")])

    result = vendor_lookup(gl, vendors)

    assert result.total_rows == 1
    assert result.unmatched_vendor_id_rows == 1
    assert result.matched_rows == 0
    assert result.no_vendor_id_rows == 0
    assert "T_UNKNOWN_VENDOR" in set(result.rows["txn_id"])
    assert pd.isna(result.rows.set_index("txn_id").loc["T_UNKNOWN_VENDOR", "vendor_name"])


def test_left_join_row_count_never_shrinks(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row(vendor_id="V1001")])
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", vendor_id="V1001"),
            _gl_row(txn_id="T_B", vendor_id=""),
            _gl_row(txn_id="T_C", vendor_id="V9999"),
        ],
    )

    result = vendor_lookup(gl, vendors)

    assert len(result.rows) == len(gl) == 3
    assert result.matched_rows + result.no_vendor_id_rows + result.unmatched_vendor_id_rows == 3


def test_missing_vendor_id_column_raises(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row()])
    gl = _load_gl(write_csv, [_gl_row()]).drop(columns=["vendor_id"])

    with pytest.raises(ValueError):
        vendor_lookup(gl, vendors)


# ---------------------------------------------------------------------------
# detect_alias_clusters: candidates only, never applied/merged.
# ---------------------------------------------------------------------------


def test_case_and_suffix_variants_form_a_cluster(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="ACME"),
            _vendor_row(vendor_id="V1003", vendor_name="Zenith Corp"),
        ],
    )

    result = detect_alias_clusters(vendors)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.normalized_name == "acme"
    assert set(cluster.vendor_ids) == {"V1001", "V1002"}
    assert set(cluster.vendor_names) == {"Acme Inc.", "ACME"}


def test_singleton_names_produce_no_cluster(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="Zenith Corp"),
        ],
    )

    result = detect_alias_clusters(vendors)

    assert result.clusters == []


def test_multiple_independent_clusters_detected(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="ACME"),
            _vendor_row(vendor_id="V1003", vendor_name="Northwind Traders"),
            _vendor_row(vendor_id="V1004", vendor_name="Northwind Traders LLC"),
        ],
    )

    result = detect_alias_clusters(vendors)

    assert {c.normalized_name for c in result.clusters} == {"acme", "northwind traders"}


def test_clusters_are_candidates_only_no_canonical_id_field(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="ACME"),
        ],
    )

    result = detect_alias_clusters(vendors)

    assert isinstance(result.clusters[0], AliasCluster)
    # A cluster names candidate vendor_ids; it never designates one as
    # canonical or produces a merged/rewritten id.
    assert not hasattr(result.clusters[0], "canonical_vendor_id")


def test_limitation_note_travels_with_every_result(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="ACME"),
        ],
    )

    result = detect_alias_clusters(vendors)

    assert "abbreviations" in result.limitation
    assert "translations" in result.limitation
    assert "canonical" in result.limitation


def test_missing_vendor_name_column_raises(write_csv):
    vendors = _load_vendors(write_csv, [_vendor_row()]).drop(columns=["vendor_name"])

    with pytest.raises(ValueError):
        detect_alias_clusters(vendors)


# ---------------------------------------------------------------------------
# Structural smoke tests against real data/ — no concrete values/counts.
# ---------------------------------------------------------------------------


def test_vendor_lookup_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    vendors = load_vendors()

    result = vendor_lookup(gl, vendors)

    assert len(result.rows) == len(gl)
    assert result.matched_rows + result.no_vendor_id_rows + result.unmatched_vendor_id_rows == result.total_rows


def test_alias_clusters_structural_smoke_against_real_data():
    vendors = load_vendors()

    result = detect_alias_clusters(vendors)

    assert result.limitation

    seen_ids = set()
    for cluster in result.clusters:
        assert len(cluster.vendor_ids) >= 2
        assert len(cluster.vendor_ids) == len(cluster.vendor_names)
        # No vendor_id appears in more than one candidate cluster.
        assert not (seen_ids & set(cluster.vendor_ids))
        seen_ids |= set(cluster.vendor_ids)
