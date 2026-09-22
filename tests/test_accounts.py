"""Tests for finance_assistant.tools.accounts.resolve_account_hierarchy (R3).

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loaders, never against the real data/*.csv
files. Account codes used here are entirely made up — never the real
dataset's mid-year-transitioning code — per R3's "no hardcodear el codigo
de esa cuenta".
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_chart_of_accounts, load_gl_transactions
from finance_assistant.tools.accounts import AccountMappingError, resolve_account_hierarchy


def _coa_row(**overrides):
    row = {
        "account_code": "9000",
        "account_name": "Synthetic Account",
        "parent_code": "9000P",
        "parent_name": "Synthetic Parent",
        "statement_line": "Operating Expenses",
        "valid_from": "2024-01-01",
        "valid_to": "2024-12-31",
    }
    row.update(overrides)
    return row


def _gl_row(**overrides):
    row = {
        "txn_id": "T0001",
        "posting_date": "2024-06-05",
        "accrual_date": "2024-06-01",
        "entity": "MI-US",
        "cost_centre": "OPS-NA",
        "account_code": "9000",
        "amount": "100.00",
        "currency": "USD",
        "vendor_id": "V1001",
        "doc_ref": "INV-0001",
        "approval_ref": "",
        "memo": "test row",
    }
    row.update(overrides)
    return row


def _load_coa(write_csv, rows):
    path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], rows)
    return load_chart_of_accounts(path)


def _load_gl(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


# ---------------------------------------------------------------------------
# Simple unique mapping.
# ---------------------------------------------------------------------------


def test_simple_unique_mapping(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001", parent_code="9100", parent_name="Parent One")])
    gl = _load_gl(write_csv, [_gl_row(account_code="9001", accrual_date="2024-06-01")])

    result = resolve_account_hierarchy(gl, coa)

    assert result.matched_rows == 1
    assert result.total_rows == 1
    assert result.is_complete
    assert result.rows["parent_code"].iloc[0] == "9100"
    assert result.rows["is_account_mapped"].iloc[0] == True  # noqa: E712 (numpy bool, not python bool)


# ---------------------------------------------------------------------------
# R3: an account code with two non-overlapping vigencia windows.
# ---------------------------------------------------------------------------


def _split_coa(write_csv):
    return _load_coa(
        write_csv,
        [
            _coa_row(
                account_code="9002",
                parent_code="9100",
                parent_name="Parent Before Split",
                valid_from="2024-01-01",
                valid_to="2024-06-30",
            ),
            _coa_row(
                account_code="9002",
                parent_code="9200",
                parent_name="Parent After Split",
                valid_from="2024-07-01",
                valid_to="9999-12-31",
            ),
        ],
    )


def test_temporal_split_maps_to_correct_parent_per_window(write_csv):
    coa = _split_coa(write_csv)
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_BEFORE", account_code="9002", accrual_date="2024-03-01"),
            _gl_row(txn_id="T_AFTER", account_code="9002", accrual_date="2024-09-01"),
        ],
    )

    result = resolve_account_hierarchy(gl, coa)

    by_txn = result.rows.set_index("txn_id")
    assert by_txn.loc["T_BEFORE", "parent_code"] == "9100"
    assert by_txn.loc["T_AFTER", "parent_code"] == "9200"


def test_date_field_choice_changes_which_window_a_row_resolves_to(write_csv):
    coa = _split_coa(write_csv)
    gl = _load_gl(
        write_csv,
        [_gl_row(account_code="9002", accrual_date="2024-03-01", posting_date="2024-09-01")],
    )

    default_result = resolve_account_hierarchy(gl, coa)
    posting_result = resolve_account_hierarchy(gl, coa, date_field="posting_date")

    assert default_result.rows["parent_code"].iloc[0] == "9100"
    assert posting_result.rows["parent_code"].iloc[0] == "9200"


# ---------------------------------------------------------------------------
# Zero-match: strict (default) raises, strict=False degrades to coverage.
# ---------------------------------------------------------------------------


def test_zero_match_raises_when_strict(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001")])
    gl = _load_gl(write_csv, [_gl_row(account_code="9999")])

    with pytest.raises(AccountMappingError) as exc_info:
        resolve_account_hierarchy(gl, coa)

    assert exc_info.value.unmapped[0].match_count == 0
    assert exc_info.value.unmapped[0].account_code == "9999"
    assert exc_info.value.ambiguous == []


def test_zero_match_degrades_to_coverage_when_not_strict(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001", parent_code="9100")])
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_MAPPED", account_code="9001"),
            _gl_row(txn_id="T_ORPHAN", account_code="9999"),
        ],
    )

    result = resolve_account_hierarchy(gl, coa, strict=False)

    assert result.is_complete is False
    assert result.matched_rows == 1
    assert result.total_rows == 2
    assert [i.account_code for i in result.unmapped] == ["9999"]

    by_txn = result.rows.set_index("txn_id")
    assert by_txn.loc["T_MAPPED", "is_account_mapped"] == True  # noqa: E712
    assert by_txn.loc["T_MAPPED", "parent_code"] == "9100"
    assert by_txn.loc["T_ORPHAN", "is_account_mapped"] == False  # noqa: E712
    assert pd.isna(by_txn.loc["T_ORPHAN", "parent_code"])


# ---------------------------------------------------------------------------
# Ambiguous match: always a hard error, regardless of strict.
# ---------------------------------------------------------------------------


def _overlapping_coa(write_csv):
    return _load_coa(
        write_csv,
        [
            _coa_row(
                account_code="9003",
                parent_code="9300",
                valid_from="2024-01-01",
                valid_to="2024-12-31",
            ),
            _coa_row(
                account_code="9003",
                parent_code="9400",
                valid_from="2024-06-01",
                valid_to="2024-12-31",
            ),
        ],
    )


@pytest.mark.parametrize("strict", [True, False])
def test_ambiguous_match_always_raises(write_csv, strict):
    coa = _overlapping_coa(write_csv)
    gl = _load_gl(write_csv, [_gl_row(account_code="9003", accrual_date="2024-07-01")])

    with pytest.raises(AccountMappingError) as exc_info:
        resolve_account_hierarchy(gl, coa, strict=strict)

    assert exc_info.value.ambiguous[0].match_count == 2
    assert exc_info.value.ambiguous[0].account_code == "9003"


def test_ambiguity_wins_over_unmapped_in_mixed_fixture(write_csv):
    coa = _overlapping_coa(write_csv)
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T_AMBIGUOUS", account_code="9003", accrual_date="2024-07-01"),
            _gl_row(txn_id="T_ORPHAN", account_code="9999"),
        ],
    )

    with pytest.raises(AccountMappingError) as exc_info:
        resolve_account_hierarchy(gl, coa, strict=False)

    assert len(exc_info.value.ambiguous) == 1
    assert len(exc_info.value.unmapped) == 1


# ---------------------------------------------------------------------------
# Sentinel 9999-12-31 handling.
# ---------------------------------------------------------------------------


def test_sentinel_valid_to_matches_far_future_dates(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9004", valid_to="9999-12-31")])
    gl = _load_gl(write_csv, [_gl_row(account_code="9004", accrual_date="2030-06-01")])

    result = resolve_account_hierarchy(gl, coa)

    assert result.matched_rows == 1


# ---------------------------------------------------------------------------
# row_id enrichment is opportunistic, not required.
# ---------------------------------------------------------------------------


def test_row_id_populated_when_txn_id_present(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001")])
    gl = _load_gl(write_csv, [_gl_row(txn_id="T_SPECIFIC", account_code="9999")])

    with pytest.raises(AccountMappingError) as exc_info:
        resolve_account_hierarchy(gl, coa)

    assert exc_info.value.unmapped[0].row_id == "T_SPECIFIC"
    assert exc_info.value.unmapped[0].row_index is not None


def test_row_id_none_when_no_txn_id_column(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001")])
    gl = _load_gl(write_csv, [_gl_row(account_code="9999")]).drop(columns=["txn_id"])

    with pytest.raises(AccountMappingError) as exc_info:
        resolve_account_hierarchy(gl, coa)

    assert exc_info.value.unmapped[0].row_id is None
    assert exc_info.value.unmapped[0].row_index == gl.index[0]


# ---------------------------------------------------------------------------
# Row-count preservation on success.
# ---------------------------------------------------------------------------


def test_success_preserves_row_count(write_csv):
    coa = _load_coa(write_csv, [_coa_row(account_code="9001")])
    gl = _load_gl(write_csv, [_gl_row(txn_id=f"T{i:04d}", account_code="9001") for i in range(3)])

    result = resolve_account_hierarchy(gl, coa)

    assert len(result.rows) == len(gl) == result.total_rows
    assert result.matched_rows == result.total_rows


# ---------------------------------------------------------------------------
# Structural smoke test against real data/.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()

    result = resolve_account_hierarchy(gl, coa, strict=True)

    assert result.is_complete is True
    assert result.matched_rows == len(gl)
