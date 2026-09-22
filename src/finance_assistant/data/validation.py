"""Deterministic schema validation and type coercion for loaded CSVs.

Consumes the declarative schemas in `finance_assistant.config`. Every failure
raises a `DataValidationError` subclass naming the offending file, never a
bare pandas exception or a silent NaN.
"""

import pandas as pd

from finance_assistant.config import DatasetSchema


class DataValidationError(Exception):
    """Base class for all dataset validation failures. Always names the file."""

    def __init__(self, file: str, message: str) -> None:
        self.file = file
        self.message = message
        super().__init__(f"[{file}] {message}")


class MissingColumnsError(DataValidationError):
    def __init__(self, file: str, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(file, f"missing required column(s): {', '.join(missing)}")


class InvalidColumnTypeError(DataValidationError):
    def __init__(self, file: str, column: str, reason: str) -> None:
        self.column = column
        self.reason = reason
        super().__init__(file, f"column '{column}' failed type validation: {reason}")


def validate_required_columns(df: pd.DataFrame, file: str, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(file, missing)


def coerce_date_column(df: pd.DataFrame, file: str, column: str) -> pd.Series:
    """Coerces to datetime64[s], not pandas' default ns resolution.

    chart_of_accounts.csv uses the sentinel 9999-12-31 for an open-ended
    valid_to, which overflows the ns-resolution Timestamp range (~year 2262).
    Applied uniformly to every date column for consistency.
    """
    try:
        return df[column].astype("datetime64[s]")
    except (ValueError, TypeError) as exc:
        raise InvalidColumnTypeError(file, column, str(exc)) from exc


def coerce_numeric_column(df: pd.DataFrame, file: str, column: str) -> pd.Series:
    coerced = pd.to_numeric(df[column], errors="coerce")
    bad_mask = coerced.isna() & df[column].notna()
    if bad_mask.any():
        bad_rows = df.index[bad_mask][:5].tolist()
        raise InvalidColumnTypeError(
            file, column, f"non-numeric value(s) at row(s) {bad_rows}"
        )
    return coerced


def normalize_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Strips whitespace, coerces to pandas string dtype, blank -> <NA>.

    Ensures blank vendor_id / approval_ref are represented consistently as
    missing, never as empty-string in one place and NaN in another.
    """
    for col in columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df


def validate_schema(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Single entry point: required-columns check, then coerces date/numeric/
    string columns per the DatasetSchema. Returns the validated DataFrame."""
    validate_required_columns(df, schema["filename"], schema["required_columns"])
    for col in schema["date_columns"]:
        df[col] = coerce_date_column(df, schema["filename"], col)
    for col in schema["numeric_columns"]:
        df[col] = coerce_numeric_column(df, schema["filename"], col)
    df = normalize_string_columns(df, schema["string_columns"])
    return df
