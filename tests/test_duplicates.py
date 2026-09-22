"""Tests for finance_assistant.tools.duplicates (R7): detect_duplicate_candidates
(HIGH/MEDIUM/LOW confidence tiers by economic fingerprint) and the credit-memo
reversal annotation.

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loaders, never against the real data/*.csv
files. Expected candidate memberships are hand-picked, never derived by
calling the function under test.
"""

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_gl_transactions, load_vendors
from finance_assistant.tools.duplicates import (
    DuplicateDetectionRules,
    detect_duplicate_candidates,
    normalize_memo_text,
)
from finance_assistant.tools.vendors import detect_alias_clusters


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
# normalize_memo_text.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Consulting Fee!", "consulting fee"),
        ("consulting  fee", "consulting fee"),
        ("Office Supplies - Q1", "office supplies q1"),
        ("hotel - 2 nights, London", "hotel 2 nights london"),
    ],
)
def test_normalize_memo_text(raw, expected):
    assert normalize_memo_text(raw) == expected


# ---------------------------------------------------------------------------
# HIGH tier: same entity + vendor_id + currency + memo + accrual_date.
# ---------------------------------------------------------------------------


def test_high_tier_exact_fingerprint_match(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", doc_ref="INV-0001", memo="office supplies"),
            _gl_row(txn_id="T_B", doc_ref="INV-0002", memo="office supplies"),
            _gl_row(txn_id="T_C", doc_ref="INV-0003", memo="a completely different memo"),
        ],
    )

    result = detect_duplicate_candidates(gl)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.confidence == "HIGH"
    assert {candidate.txn_id_a, candidate.txn_id_b} == {"T_A", "T_B"}
    assert candidate.amount == pytest.approx(100.00)


def test_high_tier_pair_never_double_counted_at_medium(write_csv):
    # Same rows as above -- same posting_date and normalized memo too, so this
    # pair would also qualify for MEDIUM if tier precedence weren't enforced.
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", doc_ref="INV-0001", memo="office supplies"),
            _gl_row(txn_id="T_B", doc_ref="INV-0002", memo="office supplies"),
        ],
    )

    result = detect_duplicate_candidates(gl)

    matching = [c for c in result.candidates if {c.txn_id_a, c.txn_id_b} == {"T_A", "T_B"}]
    assert len(matching) == 1
    assert matching[0].confidence == "HIGH"


def test_high_tier_requires_vendor_id(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", vendor_id="", memo="office supplies"),
            _gl_row(txn_id="T_B", vendor_id="", memo="office supplies"),
        ],
    )

    result = detect_duplicate_candidates(gl)

    assert result.candidates == []


# ---------------------------------------------------------------------------
# MEDIUM tier: same vendor + currency + amount + normalized memo,
# posting_date within the configurable window.
# ---------------------------------------------------------------------------


def test_medium_tier_matches_within_default_window(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", memo="Consulting Fee!", posting_date="2024-01-01", accrual_date="2024-01-01"),
            _gl_row(txn_id="T_B", memo="consulting fee", posting_date="2024-01-04", accrual_date="2024-01-04"),
        ],
    )

    result = detect_duplicate_candidates(gl)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.confidence == "MEDIUM"
    assert {candidate.txn_id_a, candidate.txn_id_b} == {"T_A", "T_B"}


def test_medium_tier_no_match_outside_default_window(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", memo="Consulting Fee!", posting_date="2024-01-01", accrual_date="2024-01-01"),
            _gl_row(txn_id="T_B", memo="consulting fee", posting_date="2024-01-11", accrual_date="2024-01-11"),
        ],
    )

    result = detect_duplicate_candidates(gl)

    assert result.candidates == []


def test_medium_tier_window_override_widens_match(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", memo="Consulting Fee!", posting_date="2024-01-01", accrual_date="2024-01-01"),
            _gl_row(txn_id="T_B", memo="consulting fee", posting_date="2024-01-11", accrual_date="2024-01-11"),
        ],
    )

    result = detect_duplicate_candidates(gl, rules=DuplicateDetectionRules(window_days=14))

    assert len(result.candidates) == 1
    assert result.candidates[0].confidence == "MEDIUM"


# ---------------------------------------------------------------------------
# LOW tier: same amount + currency + window, different vendor but same
# candidate alias cluster.
# ---------------------------------------------------------------------------


def _low_tier_gl_rows():
    return [
        _gl_row(
            txn_id="T_A",
            vendor_id="V1001",
            amount="500.00",
            memo="hardware",
            posting_date="2024-02-01",
            accrual_date="2024-02-01",
        ),
        _gl_row(
            txn_id="T_B",
            vendor_id="V1002",
            amount="500.00",
            memo="hardware purchase",
            posting_date="2024-02-03",
            accrual_date="2024-02-03",
        ),
    ]


def test_low_tier_matches_via_alias_cluster(write_csv):
    vendors = _load_vendors(
        write_csv,
        [
            _vendor_row(vendor_id="V1001", vendor_name="Acme Inc."),
            _vendor_row(vendor_id="V1002", vendor_name="ACME"),
        ],
    )
    gl = _load_gl(write_csv, _low_tier_gl_rows())
    clusters = detect_alias_clusters(vendors)

    result = detect_duplicate_candidates(gl, alias_clusters=clusters)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.confidence == "LOW"
    assert {candidate.txn_id_a, candidate.txn_id_b} == {"T_A", "T_B"}


def test_low_tier_skipped_without_alias_clusters(write_csv):
    gl = _load_gl(write_csv, _low_tier_gl_rows())

    result = detect_duplicate_candidates(gl, alias_clusters=None)

    assert result.candidates == []


# ---------------------------------------------------------------------------
# R7 reversal annotation: kept, never silently dropped.
# ---------------------------------------------------------------------------


def test_reversed_candidate_is_annotated_not_dropped(write_csv):
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_A", doc_ref="INV-1001", memo="consulting"),
            _gl_row(txn_id="T_B", doc_ref="INV-1002", memo="consulting"),
            _gl_row(
                txn_id="T_REV",
                doc_ref="CM-1001",
                amount="-100.00",
                memo="credit memo - reversal of INV-1001",
            ),
        ],
    )

    result = detect_duplicate_candidates(gl)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.is_reversed is True
    assert candidate.reversal_a is not None
    assert candidate.reversal_a.reversing_txn_id == "T_REV"
    assert candidate.reversal_a.reversing_amount == pytest.approx(-100.00)


def test_missing_required_column_raises(write_csv):
    gl = _load_gl(write_csv, [_gl_row()]).drop(columns=["memo"])

    with pytest.raises(ValueError):
        detect_duplicate_candidates(gl)


# ---------------------------------------------------------------------------
# Structural smoke tests against real data/ -- no concrete values/counts.
# ---------------------------------------------------------------------------


def test_detect_duplicate_candidates_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    vendors = load_vendors()
    clusters = detect_alias_clusters(vendors)

    result = detect_duplicate_candidates(gl, alias_clusters=clusters)

    assert result.limitation

    seen_pairs_by_tier: dict[str, set[frozenset]] = {"HIGH": set(), "MEDIUM": set(), "LOW": set()}
    all_pairs_seen: set[frozenset] = set()
    for candidate in result.candidates:
        assert candidate.txn_id_a != candidate.txn_id_b
        pair_key = frozenset({candidate.txn_id_a, candidate.txn_id_b})
        # No (txn_id_a, txn_id_b) pair appears in more than one tier.
        assert pair_key not in all_pairs_seen
        all_pairs_seen.add(pair_key)
        seen_pairs_by_tier[candidate.confidence].add(pair_key)
