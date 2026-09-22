"""Generic data-quality profiler for the Meridian datasets.

Detects anomalies by comparing datasets against each other — no dataset-
specific constant (account code, currency, entity) is hardcoded here. Prints
to stdout only; never writes to data/ or any other file. Must run unchanged
against a second dataset that shares this schema.
"""

import pandas as pd

from finance_assistant import config
from finance_assistant.data.loaders import (
    load_budget,
    load_chart_of_accounts,
    load_fx_rates,
    load_gl_transactions,
)


def report_date_ranges(gl: pd.DataFrame) -> str:
    lines = ["## Rangos de fecha (GL)"]
    for col in ["posting_date", "accrual_date"]:
        lines.append(f"- {col}: {gl[col].min().date()} .. {gl[col].max().date()}")
    return "\n".join(lines)


def report_posting_accrual_month_mismatch(gl: pd.DataFrame) -> str:
    posting_month = gl["posting_date"].dt.to_period("M")
    accrual_month = gl["accrual_date"].dt.to_period("M")
    mismatch_mask = posting_month != accrual_month
    n = int(mismatch_mask.sum())
    pct = 100 * n / len(gl) if len(gl) else 0.0
    lines = [
        "## Filas donde el mes de posting_date difiere del de accrual_date",
        f"- {n} filas ({pct:.2f}% del GL)",
    ]
    if n:
        sample = gl.loc[mismatch_mask, ["txn_id", "posting_date", "accrual_date"]].head(5)
        lines.append(f"- muestra:\n{sample.to_string(index=False)}")

    year_mismatch_mask = gl["posting_date"].dt.year != gl["accrual_date"].dt.year
    n_year = int(year_mismatch_mask.sum())
    pct_year = 100 * n_year / len(gl) if len(gl) else 0.0
    lines.append(
        "### Desglose: filas donde el AÑO de posting_date difiere del de accrual_date"
    )
    lines.append(f"- {n_year} filas ({pct_year:.2f}% del GL)")
    if n_year:
        by_currency = gl.loc[year_mismatch_mask].groupby("currency")["amount"].agg(
            ["count", "sum"]
        )
        by_currency = by_currency.rename(columns={"count": "filas", "sum": "importe_local"})
        lines.append(f"- importe agregado por moneda (local, sin convertir):\n{by_currency.to_string()}")

    return "\n".join(lines)


def report_missing_fx_combinations(gl: pd.DataFrame, fx: pd.DataFrame) -> str:
    financial_date = gl[config.DEFAULT_FINANCIAL_DATE_FIELD]
    gl_months = set(financial_date.dt.strftime("%Y-%m"))
    gl_currencies = set(gl["currency"].dropna())
    expected = {(m, c) for m in gl_months for c in gl_currencies}
    present = set(zip(fx["period_month"], fx["currency"]))
    missing = sorted(expected - present)

    lines = [
        f"## Combinaciones (mes, moneda) en GL (base {config.DEFAULT_FINANCIAL_DATE_FIELD}) ausentes en fx_rates",
        f"- {len(missing)} combinaciones faltantes de {len(expected)} esperadas",
    ]
    for month, currency in missing:
        lines.append(f"  - {month} / {currency}")
    return "\n".join(lines)


def report_multi_validity_accounts(coa: pd.DataFrame) -> str:
    counts = coa.groupby("account_code").size()
    multi = counts[counts > 1]
    lines = [
        "## account_codes con más de una fila de vigencia en el COA",
        f"- {len(multi)} account_codes con múltiples filas",
    ]
    for code, n in multi.items():
        lines.append(f"  - {code}: {n} filas")
    return "\n".join(lines)


def report_duplicate_budget_keys(budget: pd.DataFrame) -> str:
    dims = ["entity", "cost_centre", "account_code", "period_month"]
    counts = budget.groupby(dims).size()
    dup = counts[counts > 1]
    lines = [
        "## Claves dimensionales repetidas en budget (entity+cost_centre+account_code+period_month)",
        f"- {len(dup)} claves repetidas, {int(dup.sum())} filas involucradas",
    ]
    for key, n in dup.head(10).items():
        lines.append(f"  - {key}: {n} filas")
    if len(dup) > 10:
        lines.append(f"  - ... y {len(dup) - 10} claves más")
    return "\n".join(lines)


def report_missing_vendor_id(gl: pd.DataFrame) -> str:
    missing_mask = gl["vendor_id"].isna()
    n = int(missing_mask.sum())
    pct = 100 * n / len(gl) if len(gl) else 0.0
    return (
        "## Filas del GL sin vendor_id\n"
        f"- {n} filas ({pct:.2f}% del GL)"
    )


def main() -> None:
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()
    budget = load_budget()
    fx = load_fx_rates()

    sections = [
        report_date_ranges(gl),
        report_posting_accrual_month_mismatch(gl),
        report_missing_fx_combinations(gl, fx),
        report_multi_validity_accounts(coa),
        report_duplicate_budget_keys(budget),
        report_missing_vendor_id(gl),
    ]

    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
