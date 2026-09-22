"""R2 — FX conversion to USD, never interpolated, always coverage-aware.

The join key is (month of the financial date, currency). A missing rate
is never filled from a neighboring month, never assumed, never dropped
silently — it surfaces as a `MissingFXRate` and drags down `FxCoverage`.

The structural point of this module: there is no code path, scalar or
grouped, that returns a bare USD float. `aggregate_usd` and
`aggregate_usd_by` both return `UsdAmount`, which pairs a value with the
`FxCoverage` it came from and deliberately implements none of the
coercion dunders (`__float__`, `__add__`, ...) — so `sum(...)`, `float(x)`
and `x + y` all fail loudly with `TypeError` instead of silently decaying
into a number that has forgotten some rows were unconvertible.
"""

from dataclasses import dataclass

import pandas as pd

from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD


@dataclass(frozen=True)
class MissingFXRate:
    currency: str
    period_month: str
    affected_rows: int
    affected_amount_local: float


@dataclass(frozen=True)
class FxCurrencyCoverage:
    currency: str
    total_rows: int
    convertible_rows: int
    total_amount_local: float
    convertible_amount_local: float


@dataclass(frozen=True)
class FxCoverage:
    selected_rows: int
    convertible_rows: int
    per_currency: dict[str, FxCurrencyCoverage]

    @property
    def is_complete(self) -> bool:
        return self.convertible_rows == self.selected_rows


@dataclass(frozen=True)
class FxConversionResult:
    rows: pd.DataFrame
    coverage: FxCoverage
    missing: list[MissingFXRate]


class IncompleteFxCoverageError(Exception):
    def __init__(self, coverage: FxCoverage) -> None:
        self.coverage = coverage
        super().__init__(
            f"USD aggregate has incomplete FX coverage "
            f"({coverage.convertible_rows}/{coverage.selected_rows} rows convertible); "
            "call is unsafe without acknowledging coverage"
        )


@dataclass(frozen=True)
class UsdAmount:
    """No __float__/__int__/__add__/__radd__/__index__ or other coercion
    dunder on purpose: float(x), sum([x, y]), f"{x:.2f}", x + y all raise
    TypeError instead of silently decaying to a number that forgot its
    coverage. The only legitimate path to a bare float is
    require_full_coverage(), which names its own precondition."""

    converted_amount_usd: float
    coverage: FxCoverage

    def require_full_coverage(self) -> float:
        if not self.coverage.is_complete:
            raise IncompleteFxCoverageError(self.coverage)
        return self.converted_amount_usd


def convert_to_usd(
    rows: pd.DataFrame,
    fx: pd.DataFrame,
    target: str = "USD",
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> FxConversionResult:
    if target != "USD":
        raise ValueError(f"unsupported conversion target {target!r}: fx_rates.csv only provides rate_to_usd")
    if date_field not in rows.columns:
        raise ValueError(f"date_field '{date_field}' is not a column of the supplied rows")
    if not pd.api.types.is_datetime64_any_dtype(rows[date_field]):
        raise ValueError(f"date_field '{date_field}' is not a datetime column")
    for column in ("currency", "amount"):
        if column not in rows.columns:
            raise ValueError(f"rows must have a '{column}' column")

    working = rows.copy()
    working["period_month"] = working[date_field].dt.strftime("%Y-%m").astype("string")
    working["_row_index"] = working.index

    available = fx[["period_month", "currency", "rate_to_usd"]]

    present_months = working["period_month"].dropna().unique()
    present_currencies = working["currency"].dropna().unique()
    required = pd.MultiIndex.from_product(
        [present_months, present_currencies], names=["period_month", "currency"]
    ).to_frame(index=False)
    required_rates = required.merge(available, on=["period_month", "currency"], how="left")
    missing_combos = required_rates.loc[required_rates["rate_to_usd"].isna(), ["period_month", "currency"]]

    missing: list[MissingFXRate] = []
    for combo in missing_combos.to_dict("records"):
        mask = (working["period_month"] == combo["period_month"]) & (working["currency"] == combo["currency"])
        missing.append(
            MissingFXRate(
                currency=combo["currency"],
                period_month=combo["period_month"],
                affected_rows=int(mask.sum()),
                affected_amount_local=float(working.loc[mask, "amount"].sum()),
            )
        )

    merged = working.merge(available, on=["period_month", "currency"], how="left")
    merged["is_fx_convertible"] = merged["rate_to_usd"].notna()
    merged["amount_usd"] = merged["amount"] * merged["rate_to_usd"]
    merged = merged.set_index("_row_index").sort_index()
    merged.index.name = rows.index.name
    result_rows = merged.drop(columns=["rate_to_usd"])

    per_currency = {
        ccy: FxCurrencyCoverage(
            currency=ccy,
            total_rows=len(group),
            convertible_rows=int(group["is_fx_convertible"].sum()),
            total_amount_local=float(group["amount"].sum()),
            convertible_amount_local=float(group.loc[group["is_fx_convertible"], "amount"].sum()),
        )
        for ccy, group in result_rows.groupby("currency", dropna=False)
    }
    coverage = FxCoverage(
        selected_rows=len(result_rows),
        convertible_rows=int(result_rows["is_fx_convertible"].sum()),
        per_currency=per_currency,
    )

    return FxConversionResult(rows=result_rows, coverage=coverage, missing=missing)


def aggregate_usd(result: FxConversionResult) -> UsdAmount:
    convertible = result.rows.loc[result.rows["is_fx_convertible"]]
    return UsdAmount(
        converted_amount_usd=float(convertible["amount_usd"].sum()),
        coverage=result.coverage,
    )


def aggregate_usd_by(result: FxConversionResult, by: list[str]) -> dict[tuple, UsdAmount]:
    out: dict[tuple, UsdAmount] = {}
    for key, group in result.rows.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        convertible = group.loc[group["is_fx_convertible"]]
        per_currency = {
            ccy: FxCurrencyCoverage(
                currency=ccy,
                total_rows=len(sub),
                convertible_rows=int(sub["is_fx_convertible"].sum()),
                total_amount_local=float(sub["amount"].sum()),
                convertible_amount_local=float(sub.loc[sub["is_fx_convertible"], "amount"].sum()),
            )
            for ccy, sub in group.groupby("currency", dropna=False)
        }
        group_coverage = FxCoverage(
            selected_rows=len(group),
            convertible_rows=int(group["is_fx_convertible"].sum()),
            per_currency=per_currency,
        )
        out[key] = UsdAmount(
            converted_amount_usd=float(convertible["amount_usd"].sum()),
            coverage=group_coverage,
        )
    return out
