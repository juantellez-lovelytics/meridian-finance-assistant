"""Tests for finance_assistant.tools.fx.convert_to_usd (R2).

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loaders, never against the real data/*.csv
files. Expected amounts are always hand-computed, never derived by
calling the function under test.
"""

import pandas as pd
import pytest

from finance_assistant import config
from finance_assistant.data.loaders import load_fx_rates, load_gl_transactions
from finance_assistant.tools.fx import (
    FxConversionResult,
    IncompleteFxCoverageError,
    MissingFXRate,
    aggregate_usd,
    aggregate_usd_by,
    convert_to_usd,
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


def _fx_row(**overrides):
    row = {"period_month": "2024-01", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _load_gl(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


def _load_fx(write_csv, rows):
    path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], rows)
    return load_fx_rates(path)


# ---------------------------------------------------------------------------
# Exact hand-computed arithmetic.
# ---------------------------------------------------------------------------


def test_exact_arithmetic(write_csv):
    fx = _load_fx(write_csv, [_fx_row(period_month="2024-01", currency="EUR", rate_to_usd="1.1")])
    gl = _load_gl(write_csv, [_gl_row(accrual_date="2024-01-15", currency="EUR", amount="100")])

    result = convert_to_usd(gl, fx)

    assert result.rows["amount_usd"].iloc[0] == pytest.approx(110.0)
    assert bool(result.rows["is_fx_convertible"].iloc[0])


def test_usd_passthrough_uses_the_same_code_path(write_csv):
    fx = _load_fx(write_csv, [_fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0")])
    gl = _load_gl(write_csv, [_gl_row(accrual_date="2024-01-10", currency="USD", amount="250")])

    result = convert_to_usd(gl, fx)

    assert result.rows["amount_usd"].iloc[0] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Single missing (period_month, currency) combo, cartesian-product detection.
# ---------------------------------------------------------------------------


def test_single_missing_combo_detected_via_cartesian_product(write_csv):
    fx = _load_fx(
        write_csv,
        [
            _fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0"),
            _fx_row(period_month="2024-01", currency="EUR", rate_to_usd="1.1"),
            _fx_row(period_month="2024-02", currency="USD", rate_to_usd="1.0"),
            # 2024-02 / EUR intentionally absent.
        ],
    )
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T1", accrual_date="2024-01-01", currency="USD", amount="100"),
            _gl_row(txn_id="T2", accrual_date="2024-01-01", currency="EUR", amount="200"),
            _gl_row(txn_id="T3", accrual_date="2024-02-01", currency="USD", amount="300"),
            _gl_row(txn_id="T4", accrual_date="2024-02-01", currency="EUR", amount="400"),
        ],
    )

    result = convert_to_usd(gl, fx)

    assert result.missing == [
        MissingFXRate(currency="EUR", period_month="2024-02", affected_rows=1, affected_amount_local=400.0)
    ]
    assert result.coverage.selected_rows == 4
    assert result.coverage.convertible_rows == 3

    by_txn = result.rows.set_index("txn_id")
    assert by_txn.loc["T1", "amount_usd"] == pytest.approx(100.0)
    assert by_txn.loc["T2", "amount_usd"] == pytest.approx(220.0)
    assert by_txn.loc["T3", "amount_usd"] == pytest.approx(300.0)
    assert pd.isna(by_txn.loc["T4", "amount_usd"])
    assert bool(by_txn.loc["T4", "is_fx_convertible"]) is False


# ---------------------------------------------------------------------------
# Never interpolates.
# ---------------------------------------------------------------------------


def test_never_interpolates_a_missing_month(write_csv):
    fx = _load_fx(
        write_csv,
        [
            _fx_row(period_month="2024-01", currency="GBP", rate_to_usd="1.25"),
            _fx_row(period_month="2024-03", currency="GBP", rate_to_usd="1.30"),
            # 2024-02 / GBP intentionally absent.
        ],
    )
    gl = _load_gl(write_csv, [_gl_row(accrual_date="2024-02-15", currency="GBP", amount="100")])

    result = convert_to_usd(gl, fx)

    # NaN, not silently filled from January's or March's rate.
    assert pd.isna(result.rows["amount_usd"].iloc[0])


# ---------------------------------------------------------------------------
# Per-currency coverage, hand-computed.
# ---------------------------------------------------------------------------


def test_per_currency_coverage_hand_computed(write_csv):
    fx = _load_fx(
        write_csv,
        [
            _fx_row(period_month="2024-01", currency="AUD", rate_to_usd="0.65"),
            _fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0"),
            _fx_row(period_month="2024-02", currency="USD", rate_to_usd="1.0"),
            # 2024-02 / AUD intentionally absent -> AUD partially covered.
        ],
    )
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="A1", accrual_date="2024-01-01", currency="AUD", amount="100"),
            _gl_row(txn_id="A2", accrual_date="2024-02-01", currency="AUD", amount="50"),
            _gl_row(txn_id="U1", accrual_date="2024-01-01", currency="USD", amount="10"),
            _gl_row(txn_id="U2", accrual_date="2024-02-01", currency="USD", amount="20"),
        ],
    )

    result = convert_to_usd(gl, fx)

    aud = result.coverage.per_currency["AUD"]
    assert aud.total_rows == 2
    assert aud.convertible_rows == 1
    assert aud.total_amount_local == pytest.approx(150.0)
    assert aud.convertible_amount_local == pytest.approx(100.0)

    usd = result.coverage.per_currency["USD"]
    assert usd.total_rows == 2
    assert usd.convertible_rows == 2
    assert usd.total_amount_local == pytest.approx(30.0)
    assert usd.convertible_amount_local == pytest.approx(30.0)

    assert result.coverage.selected_rows == 4
    assert result.coverage.convertible_rows == 3
    assert result.coverage.is_complete is False


# ---------------------------------------------------------------------------
# Unsupported target.
# ---------------------------------------------------------------------------


def test_non_usd_target_rejected(write_csv):
    fx = _load_fx(write_csv, [_fx_row()])
    gl = _load_gl(write_csv, [_gl_row()])

    with pytest.raises(ValueError):
        convert_to_usd(gl, fx, target="EUR")


# ---------------------------------------------------------------------------
# Scalar-decay is structurally blocked.
# ---------------------------------------------------------------------------


def test_scalar_decay_blocked_on_partial_coverage(write_csv):
    fx = _load_fx(write_csv, [_fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0")])
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T1", accrual_date="2024-01-01", currency="USD", amount="100"),
            _gl_row(txn_id="T2", accrual_date="2024-02-01", currency="USD", amount="50"),
        ],
    )

    result = convert_to_usd(gl, fx)
    usd_amount = aggregate_usd(result)

    assert result.coverage.is_complete is False
    with pytest.raises(IncompleteFxCoverageError):
        usd_amount.require_full_coverage()
    with pytest.raises(TypeError):
        float(usd_amount)
    with pytest.raises(TypeError):
        usd_amount + usd_amount


def test_require_full_coverage_returns_float_when_complete(write_csv):
    fx = _load_fx(write_csv, [_fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0")])
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T1", accrual_date="2024-01-01", currency="USD", amount="100"),
            _gl_row(txn_id="T2", accrual_date="2024-01-01", currency="USD", amount="50"),
        ],
    )

    result = convert_to_usd(gl, fx)
    usd_amount = aggregate_usd(result)

    assert result.coverage.is_complete is True
    assert usd_amount.require_full_coverage() == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# aggregate_usd_by: per-group coverage (mandatory).
# ---------------------------------------------------------------------------


def test_aggregate_usd_by_computes_independent_coverage_per_group(write_csv):
    fx = _load_fx(
        write_csv,
        [
            _fx_row(period_month="2024-01", currency="USD", rate_to_usd="1.0"),
            _fx_row(period_month="2024-01", currency="EUR", rate_to_usd="1.1"),
            _fx_row(period_month="2024-02", currency="USD", rate_to_usd="1.0"),
            # 2024-02 / EUR intentionally absent -> hits only CC-A's row.
        ],
    )
    gl = _load_gl(
        write_csv,
        [
            _gl_row(txn_id="T1", cost_centre="CC-A", accrual_date="2024-01-01", currency="USD", amount="100"),
            _gl_row(txn_id="T2", cost_centre="CC-A", accrual_date="2024-02-01", currency="EUR", amount="50"),
            _gl_row(txn_id="T3", cost_centre="CC-B", accrual_date="2024-01-01", currency="EUR", amount="200"),
            _gl_row(txn_id="T4", cost_centre="CC-B", accrual_date="2024-02-01", currency="USD", amount="300"),
        ],
    )

    result = convert_to_usd(gl, fx)
    by_cost_centre = aggregate_usd_by(result, by=["cost_centre"])

    cc_a = by_cost_centre[("CC-A",)]
    cc_b = by_cost_centre[("CC-B",)]

    assert cc_a.coverage.is_complete is False
    assert cc_b.coverage.is_complete is True

    with pytest.raises(IncompleteFxCoverageError):
        cc_a.require_full_coverage()
    assert cc_b.require_full_coverage() == pytest.approx(220.0 + 300.0)

    # CC-A's converted amount only sums its one convertible row (100 * 1.0),
    # never a NaN/0 stand-in for the row with the missing rate.
    assert cc_a.converted_amount_usd == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    fx = load_fx_rates()

    result: FxConversionResult = convert_to_usd(gl, fx)

    affected_total = sum(m.affected_rows for m in result.missing)
    assert result.coverage.convertible_rows + affected_total == result.coverage.selected_rows
