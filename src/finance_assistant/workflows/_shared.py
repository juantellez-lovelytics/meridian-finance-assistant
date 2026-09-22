"""Internal helpers shared across workflows/*.py.

Not one of the eight workflows and not exposed to the model — pure
plumbing so the year-ambiguity mechanism (decided once, applied by both
opex_by_cost_centre and consolidated_spend) isn't reinvented twice, and so
tool-call recording (ToolTrace) isn't hand-built call by call in every
workflow.
"""

import inspect
import time
from typing import Callable, TypeVar

import pandas as pd

from finance_assistant.evidence.models import AnswerStatus, ToolCall
from finance_assistant.evidence.summarize import summarize_argument_for_trace, summarize_for_trace

T = TypeVar("T")


class ToolTrace:
    """Accumulates `ToolCall` entries for one workflow invocation.

    The single place that decides what gets recorded when a `tools/*`
    function is called: binds arguments by signature (real parameter names,
    including defaults the caller didn't pass explicitly), times the call,
    and summarizes both sides. Never rebuilt by hand at each call site — a
    workflow just wraps `tt.call(some_tool, *args, **kwargs)` in place of
    `some_tool(*args, **kwargs)`.

    Arguments are summarized via `summarize_argument_for_trace`, not
    `summarize_for_trace`: a dataclass-typed argument in this codebase is
    always the object a previous tool call already returned (and therefore
    already fully described in that earlier step's own result_summary), so
    it collapses to a compact reference here instead of re-expanding a
    structure the trace already showed once. The result itself still gets
    full detail via `summarize_for_trace` — it's the new information this
    step establishes.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def call(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments_summary = {name: summarize_argument_for_trace(value) for name, value in bound.arguments.items()}

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration_ms = int((time.perf_counter() - start) * 1000)

        summarized_result = summarize_for_trace(result)
        result_summary = summarized_result if isinstance(summarized_result, dict) else {"value": summarized_result}

        self.calls.append(
            ToolCall(
                tool=fn.__name__,
                arguments_summary=arguments_summary,
                result_summary=result_summary,
                duration_ms=duration_ms,
            )
        )
        return result

_QUARTER_MONTHS = {
    "Q1": (1, 3),
    "Q2": (4, 6),
    "Q3": (7, 9),
    "Q4": (10, 12),
}


def quarter_bounds(year: int, quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if quarter not in _QUARTER_MONTHS:
        raise ValueError(f"quarter must be one of {sorted(_QUARTER_MONTHS)}, got {quarter!r}")
    start_month, end_month = _QUARTER_MONTHS[quarter]
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    return start, end


def resolve_year_or_readings(
    gl: pd.DataFrame,
    quarter: str,
    year: int | None,
    compute_fn: Callable[[int], tuple[AnswerStatus | None, float | None, dict]],
    date_field: str,
) -> tuple[int, dict, list[tuple[str, AnswerStatus | None, float | None]] | None]:
    """Business rule (decided, not reinterpretable): if `year` is omitted,
    compute a reading for every year present in gl[date_field], hand all of
    them to the gate as period_readings, and — should the gate not force
    NEEDS_CLARIFICATION — proceed with the most recent year. The caller is
    responsible for declaring that default as an assumption in its own
    EvidenceBundle, since only it knows the domain-appropriate wording.

    compute_fn(y) must run the workflow's own tool pipeline for year y and
    return (status_hint, best_effort_magnitude, payload) where payload is
    whatever the caller needs later to build its `result` dict for that
    year. status_hint/magnitude may be None only when year y genuinely has
    no matching rows at all.

    Returns (chosen_year, payload_for_chosen_year, period_readings) —
    period_readings is None when `year` was given explicitly (a single
    deterministic read never needs the ambiguous-period gate row).
    """
    if year is not None:
        _, _, payload = compute_fn(year)
        return year, payload, None

    years = sorted(int(y) for y in gl[date_field].dropna().dt.year.unique())
    if not years:
        raise ValueError(f"no years present in gl['{date_field}']")

    readings: list[tuple[str, AnswerStatus | None, float | None]] = []
    payload_by_year: dict[int, dict] = {}
    for y in years:
        status_hint, magnitude, payload = compute_fn(y)
        readings.append((f"FY{y} {quarter}", status_hint, magnitude))
        payload_by_year[y] = payload

    chosen_year = years[-1]
    return chosen_year, payload_by_year[chosen_year], readings
