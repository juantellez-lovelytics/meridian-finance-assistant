"""A pure, readable-summary formatter shared by trace instrumentation.

`workflows/_shared.py::ToolTrace` uses this to summarize tool arguments and
results when recording a `ToolCall`; `evidence/trace.py` uses it to
summarize `CalcStep` inputs/output when building `steps[]`. One
implementation, no dependency in either direction between `workflows/` and
`evidence/` beyond this module.

Never dumps raw rows: a `DataFrame` becomes its row count, long dicts/lists
are truncated with a count marker, long strings are truncated with an
ellipsis. Structured values (dataclasses, pydantic models) are summarized
recursively so a caller never has to hand-write a summary for each tool's
bespoke result type.

Two entry points, both generic (neither hardcodes a tool's dataclass name):

- `summarize_for_trace` -- full recursive detail. Used for a call's own
  *result*, since that's the new information a step establishes.
- `summarize_argument_for_trace` -- same, except a dataclass instance
  collapses to a compact one-line reference (class name + its scalar/
  DataFrame-row-count/collection-length fields, one level into any nested
  dataclass field). Used for a call's *arguments*: in this codebase, a
  dataclass-typed argument is always the very object a previous tool call
  already returned (and therefore already fully described in that earlier
  step's result_summary) -- expanding it again here would just duplicate
  that structure and inflate the trace.
"""

import dataclasses
from typing import Any

import pandas as pd
from pydantic import BaseModel

_DEFAULT_MAX_ITEMS = 8
_DEFAULT_MAX_STR_LEN = 200
_DEFAULT_COMPACT_DEPTH = 2


def _stringify_key(key: Any) -> str:
    if isinstance(key, tuple):
        if len(key) == 1:
            return str(key[0])
        return "|".join(str(k) for k in key)
    return str(key)


def summarize_for_trace(value: Any, max_items: int = _DEFAULT_MAX_ITEMS, max_str_len: int = _DEFAULT_MAX_STR_LEN) -> Any:
    if isinstance(value, pd.DataFrame):
        return len(value)

    if isinstance(value, pd.Timestamp):
        return str(value)

    try:
        na_result = pd.isna(value)
    except (TypeError, ValueError):
        na_result = False
    if pd.api.types.is_scalar(na_result) and bool(na_result):
        return None

    if isinstance(value, BaseModel):
        return summarize_for_trace(value.model_dump(), max_items, max_str_len)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: summarize_for_trace(getattr(value, f.name), max_items, max_str_len)
            for f in dataclasses.fields(value)
        }

    if isinstance(value, dict):
        items = list(value.items())
        out = {_stringify_key(k): summarize_for_trace(v, max_items, max_str_len) for k, v in items[:max_items]}
        if len(items) > max_items:
            out["__truncated__"] = f"+{len(items) - max_items} more"
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out = [summarize_for_trace(v, max_items, max_str_len) for v in items[:max_items]]
        if len(items) > max_items:
            out.append(f"+{len(items) - max_items} more")
        return out

    if isinstance(value, str) and len(value) > max_str_len:
        return value[: max_str_len - 1] + "…"

    return value


def _compact_parts(value: Any, max_items: int, max_str_len: int, depth: int) -> list[str]:
    if depth <= 0 or not (dataclasses.is_dataclass(value) and not isinstance(value, type)):
        return []
    parts: list[str] = []
    for f in dataclasses.fields(value):
        fv = getattr(value, f.name)
        if isinstance(fv, pd.DataFrame):
            parts.append(f"{len(fv)} rows")
        elif dataclasses.is_dataclass(fv) and not isinstance(fv, type):
            parts.extend(_compact_parts(fv, max_items, max_str_len, depth - 1))
        elif isinstance(fv, (list, tuple, set, frozenset, dict)):
            parts.append(f"{f.name}={len(fv)}")
        else:
            parts.append(f"{f.name}={summarize_for_trace(fv, max_items, max_str_len)}")
    return parts


def _compact_reference(value: Any, max_items: int, max_str_len: int) -> str:
    class_name = type(value).__name__
    parts = _compact_parts(value, max_items, max_str_len, _DEFAULT_COMPACT_DEPTH)
    return f"<{class_name}: {', '.join(parts)}>" if parts else f"<{class_name}>"


def summarize_argument_for_trace(value: Any, max_items: int = _DEFAULT_MAX_ITEMS, max_str_len: int = _DEFAULT_MAX_STR_LEN) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _compact_reference(value, max_items, max_str_len)
    return summarize_for_trace(value, max_items, max_str_len)
