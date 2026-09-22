"""Tests for finance_assistant.evidence.summarize.summarize_for_trace.

Pure unit tests against hand-built values -- no dataset dependency.
"""

from dataclasses import dataclass

import pandas as pd
import pytest

from finance_assistant.evidence.summarize import summarize_argument_for_trace, summarize_for_trace


def test_short_dict_passes_through_unchanged():
    assert summarize_for_trace({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_long_dict_truncated_with_count_marker():
    value = {f"k{i}": i for i in range(12)}

    result = summarize_for_trace(value, max_items=8)

    assert len(result) == 9  # 8 real keys + the truncation marker
    assert result["__truncated__"] == "+4 more"
    assert set(result) - {"__truncated__"} == {f"k{i}" for i in range(8)}


def test_short_list_passes_through_unchanged():
    assert summarize_for_trace([1, 2, 3]) == [1, 2, 3]


def test_long_list_truncated_with_count_marker():
    value = list(range(12))

    result = summarize_for_trace(value, max_items=8)

    assert result[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert result[8] == "+4 more"
    assert len(result) == 9


def test_long_string_truncated_with_ellipsis():
    value = "x" * 250

    result = summarize_for_trace(value, max_str_len=200)

    assert len(result) == 200
    assert result.endswith("…")


def test_short_string_passes_through_unchanged():
    assert summarize_for_trace("hello") == "hello"


def test_dataframe_summarized_to_row_count():
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert summarize_for_trace(df) == 3


def test_timestamp_summarized_to_string():
    ts = pd.Timestamp("2024-07-01")

    assert summarize_for_trace(ts) == str(ts)


@pytest.mark.parametrize("value", [pd.NaT, float("nan"), None, pd.NA])
def test_na_like_scalars_summarized_to_none(value):
    assert summarize_for_trace(value) is None


def test_nested_dataclass_summarized_recursively():
    @dataclass(frozen=True)
    class Inner:
        x: int
        y: str

    @dataclass(frozen=True)
    class Outer:
        inner: Inner
        total: float

    outer = Outer(inner=Inner(x=1, y="hi"), total=3.5)

    assert summarize_for_trace(outer) == {"inner": {"x": 1, "y": "hi"}, "total": 3.5}


def test_dataclass_field_holding_a_dataframe_is_row_counted():
    @dataclass(frozen=True)
    class Holder:
        rows: pd.DataFrame
        total_rows: int

    holder = Holder(rows=pd.DataFrame({"a": [1, 2]}), total_rows=2)

    assert summarize_for_trace(holder) == {"rows": 2, "total_rows": 2}


def test_scalar_passes_through_unchanged():
    assert summarize_for_trace(42) == 42
    assert summarize_for_trace(3.5) == 3.5
    assert summarize_for_trace(True) is True


def test_dict_with_single_element_tuple_keys_uses_the_bare_element():
    value = {("MI-CA",): 1.0, ("MI-NL",): 2.0}

    result = summarize_for_trace(value)

    assert result == {"MI-CA": 1.0, "MI-NL": 2.0}


def test_dict_with_multi_element_tuple_keys_joins_them():
    value = {("MI-CA", "AC1"): 1.0}

    result = summarize_for_trace(value)

    assert result == {"MI-CA|AC1": 1.0}


def test_summarize_argument_collapses_dataclass_to_compact_reference():
    @dataclass(frozen=True)
    class Coverage:
        selected_rows: int
        convertible_rows: int

    @dataclass(frozen=True)
    class Result:
        rows: pd.DataFrame
        coverage: Coverage
        missing: list

    value = Result(rows=pd.DataFrame({"a": range(1348)}), coverage=Coverage(selected_rows=1348, convertible_rows=1201), missing=[])

    result = summarize_argument_for_trace(value)

    assert isinstance(result, str)
    assert result.startswith("<Result: ")
    assert "1348 rows" in result
    assert "convertible_rows=1201" in result
    assert "missing=0" in result


def test_summarize_argument_does_not_expand_nested_lists_or_dicts():
    @dataclass(frozen=True)
    class Big:
        items: list

    value = Big(items=list(range(100)))

    result = summarize_argument_for_trace(value)

    assert result == "<Big: items=100>"
    assert "99" not in result


def test_summarize_argument_truncates_long_string_fields():
    @dataclass(frozen=True)
    class HasLongText:
        note: str

    value = HasLongText(note="x" * 250)

    result = summarize_argument_for_trace(value)

    assert "x" * 250 not in result
    assert "…" in result


def test_summarize_argument_leaves_non_dataclass_values_unchanged():
    assert summarize_argument_for_trace({"a": 1}) == {"a": 1}
    assert summarize_argument_for_trace([1, 2, 3]) == [1, 2, 3]
    assert summarize_argument_for_trace("hello") == "hello"
    assert summarize_argument_for_trace(None) is None


def test_summarize_argument_on_plain_dataframe_still_row_counts():
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert summarize_argument_for_trace(df) == 3
