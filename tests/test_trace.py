"""Tests for finance_assistant.evidence.trace (RunTrace, build_trace, write_trace).

Bundles here are built directly from the pydantic evidence models -- never
by calling a workflow -- so these tests exercise the trace builder in
isolation from any dataset.
"""

import json
from datetime import datetime, timezone

import pytest

from finance_assistant.evidence.models import (
    AnswerStatus,
    CalcStep,
    Coverage,
    EvidenceBundle,
    ToolCall,
)
from finance_assistant.evidence.trace import ModelCall, build_trace, write_trace
from finance_assistant.orchestration.intents import Intent

_STARTED_AT = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
_COVERAGE = Coverage(selected_rows=10, computable_rows=10, computable_amount_pct=100.0)


def _bundle(**overrides) -> EvidenceBundle:
    fields = {
        "status": AnswerStatus.ANSWER,
        "intent": Intent.CONSOLIDATED_SPEND,
        "result": {"total_usd": 100.0},
        "coverage": _COVERAGE,
    }
    fields.update(overrides)
    return EvidenceBundle(**fields)


def test_steps_built_from_tool_calls_then_calculations_in_order():
    bundle = _bundle(
        tool_calls=[
            ToolCall(tool="query_ledger", arguments_summary={"rows": 10}, result_summary={"rows_matched": 10}, duration_ms=5),
            ToolCall(tool="convert_to_usd", arguments_summary={"rows": 10}, result_summary={"convertible_rows": 10}, duration_ms=12),
        ],
        calculations=[
            CalcStep(description="computable USD total", operation="sum", inputs={"selected_rows": 10}, output=100.0),
        ],
    )

    trace = build_trace(question="q", bundle=bundle, started_at=_STARTED_AT, duration_ms=50, date_basis="accrual_date")

    assert [s.step for s in trace.steps] == [1, 2, 3]
    assert [s.type for s in trace.steps] == ["tool", "tool", "calc"]
    assert trace.steps[0].name == "query_ledger"
    assert trace.steps[0].duration_ms == 5
    assert trace.steps[1].name == "convert_to_usd"
    assert trace.steps[1].duration_ms == 12
    assert trace.steps[2].name == "computable USD total"
    assert trace.steps[2].duration_ms == 0
    assert trace.steps[2].arguments["operation"] == "sum"
    assert trace.steps[2].result_summary["output"] == 100.0


def test_steps_empty_when_bundle_has_no_tool_calls_or_calculations():
    bundle = _bundle()

    trace = build_trace(question="q", bundle=bundle, started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    assert trace.steps == []


def test_calc_step_arguments_and_output_are_truncated_for_readability():
    long_output = {f"cc{i}": float(i) for i in range(20)}
    bundle = _bundle(
        calculations=[
            CalcStep(description="by cost centre", operation="groupby_sum", inputs={"by": "cost_centre"}, output=long_output),
        ],
    )

    trace = build_trace(question="q", bundle=bundle, started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    step = trace.steps[0]
    output_summary = step.result_summary["output"]
    assert "__truncated__" in output_summary
    assert len(output_summary) == 9  # 8 sampled keys + the truncation marker


def test_final_evidence_is_full_bundle_untouched():
    long_output = {f"cc{i}": float(i) for i in range(20)}
    bundle = _bundle(
        calculations=[CalcStep(description="by cost centre", operation="groupby_sum", inputs={}, output=long_output)],
    )

    trace = build_trace(question="q", bundle=bundle, started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    assert trace.final_evidence == bundle.model_dump(mode="json")
    # Unlike steps[], final_evidence keeps the calculation's output untruncated.
    assert len(trace.final_evidence["calculations"][0]["output"]) == 20


def test_model_calls_default_to_empty_and_cost_is_a_zero_float_not_unknown():
    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    assert trace.model_calls == []
    assert trace.estimated_cost_usd == 0.0
    assert isinstance(trace.estimated_cost_usd, float)


def _model_call(**overrides) -> ModelCall:
    fields = {
        "provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-5",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "estimated_cost_usd": 0.01,
        "latency_ms": 250,
    }
    fields.update(overrides)
    return ModelCall(**fields)


def test_build_trace_populates_model_calls_and_sums_known_cost():
    calls = [_model_call(estimated_cost_usd=0.01), _model_call(estimated_cost_usd=0.02)]

    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date", model_calls=calls)

    assert trace.model_calls == calls
    assert trace.estimated_cost_usd == pytest.approx(0.03)


def test_build_trace_cost_is_unknown_if_any_call_cost_is_unknown():
    calls = [_model_call(estimated_cost_usd=0.01), _model_call(estimated_cost_usd="unknown")]

    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date", model_calls=calls)

    assert trace.estimated_cost_usd == "unknown"


def test_build_trace_model_calls_none_matches_omitted_default():
    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date", model_calls=None)

    assert trace.model_calls == []
    assert trace.estimated_cost_usd == 0.0


def test_status_and_date_basis_round_trip_from_bundle_and_caller():
    bundle = _bundle(status=AnswerStatus.PARTIAL, coverage=Coverage(selected_rows=10, computable_rows=8, computable_amount_pct=80.0))

    trace = build_trace(question="q", bundle=bundle, started_at=_STARTED_AT, duration_ms=10, date_basis="posting_date")

    assert trace.status == AnswerStatus.PARTIAL
    assert trace.date_basis == "posting_date"


def test_run_id_prefix_is_deterministic_function_of_started_at_and_intent():
    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    assert trace.run_id.startswith("20240701T120000Z_consolidated_spend_")


def test_run_id_unique_across_two_traces_with_identical_started_at(tmp_path):
    trace_a = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")
    trace_b = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    assert trace_a.run_id != trace_b.run_id

    path_a = write_trace(trace_a, tmp_path)
    path_b = write_trace(trace_b, tmp_path)

    assert path_a != path_b
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_write_trace_produces_valid_json_that_round_trips(tmp_path):
    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    path = write_trace(trace, tmp_path)

    assert path == tmp_path / f"{trace.run_id}.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == trace.model_dump(mode="json")


def test_write_trace_creates_traces_dir_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    trace = build_trace(question="q", bundle=_bundle(), started_at=_STARTED_AT, duration_ms=10, date_basis="accrual_date")

    path = write_trace(trace, nested)

    assert path.exists()
