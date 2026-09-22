"""Read-only CSV loaders. Never write to `data/`.

Each loader accepts an optional `path` override so tests and callers can
point at a synthetic CSV (e.g. a pytest tmp_path fixture) instead of the
real dataset.
"""

from pathlib import Path

import pandas as pd

from finance_assistant import config
from finance_assistant.data.validation import validate_schema


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_gl_transactions(path: str | Path | None = None) -> pd.DataFrame:
    resolved = Path(path) if path is not None else config.DATA_DIR / config.GL_TRANSACTIONS_FILE
    df = _read_csv(resolved)
    return validate_schema(df, config.GL_SCHEMA)


def load_chart_of_accounts(path: str | Path | None = None) -> pd.DataFrame:
    resolved = Path(path) if path is not None else config.DATA_DIR / config.CHART_OF_ACCOUNTS_FILE
    df = _read_csv(resolved)
    return validate_schema(df, config.COA_SCHEMA)


def load_budget(path: str | Path | None = None) -> pd.DataFrame:
    resolved = Path(path) if path is not None else config.DATA_DIR / config.BUDGET_FILE
    df = _read_csv(resolved)
    return validate_schema(df, config.BUDGET_SCHEMA)


def load_fx_rates(path: str | Path | None = None) -> pd.DataFrame:
    resolved = Path(path) if path is not None else config.DATA_DIR / config.FX_RATES_FILE
    df = _read_csv(resolved)
    return validate_schema(df, config.FX_SCHEMA)


def load_vendors(path: str | Path | None = None) -> pd.DataFrame:
    resolved = Path(path) if path is not None else config.DATA_DIR / config.VENDORS_FILE
    df = _read_csv(resolved)
    return validate_schema(df, config.VENDORS_SCHEMA)
