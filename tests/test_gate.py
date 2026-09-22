"""Tests for finance_assistant.evidence.gate.apply_gate.

One test per degradation-table row plus precedence/combination cases.
Everything here is synthetic in-memory Coverage/tuples — no dataset
dependency, since the gate never sees dataset content, only typed facts a
workflow already computed.
"""

from finance_assistant.evidence.gate import apply_gate, readings_are_material_difference
from finance_assistant.evidence.models import AnswerStatus, Coverage

_FULL = Coverage(selected_rows=10, computable_rows=10, computable_amount_pct=100.0)
_PARTIAL_COVERAGE = Coverage(selected_rows=10, computable_rows=8, computable_amount_pct=80.0)


def test_gate_is_a_noop_when_nothing_fires():
    result = apply_gate(draft_status=AnswerStatus.ANSWER)

    assert result.final_status == AnswerStatus.ANSWER
    assert result.warnings_added == []
    assert result.refusal_reason is None
    assert result.clarification_options == []
    assert result.firing_conditions == []


# ---------------------------------------------------------------------------
# Row A — missing FX rate affects the requested total.
# ---------------------------------------------------------------------------


def test_row_a_missing_fx_defaults_to_partial():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, missing_fx_affects_requested_total=True)

    assert result.final_status == AnswerStatus.PARTIAL
    assert result.refusal_reason is None


def test_row_a_missing_fx_escalates_to_refused_with_force_flag():
    result = apply_gate(
        draft_status=AnswerStatus.ANSWER,
        missing_fx_affects_requested_total=True,
        force_refuse_incomplete_total=True,
    )

    assert result.final_status == AnswerStatus.REFUSED
    assert result.refusal_reason is not None
    assert "FX rate" in result.refusal_reason


# ---------------------------------------------------------------------------
# Row B — missing denominator.
# ---------------------------------------------------------------------------


def test_row_b_missing_denominator_forces_refused():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, missing_denominator=True)

    assert result.final_status == AnswerStatus.REFUSED
    assert result.refusal_reason is not None


# ---------------------------------------------------------------------------
# Row C — row coverage < 100% on a consolidated total.
# ---------------------------------------------------------------------------


def test_row_c_partial_row_coverage_caps_answer_at_partial():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, row_coverage=_PARTIAL_COVERAGE)

    assert result.final_status == AnswerStatus.PARTIAL


def test_row_c_full_row_coverage_does_not_degrade_answer():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, row_coverage=_FULL)

    assert result.final_status == AnswerStatus.ANSWER


def test_row_c_fx_coverage_also_drives_the_ceiling_independently():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, fx_coverage=_PARTIAL_COVERAGE)

    assert result.final_status == AnswerStatus.PARTIAL
    assert "fx_coverage_incomplete" in result.firing_conditions


# ---------------------------------------------------------------------------
# Row D — ambiguous period, materially different readings.
# ---------------------------------------------------------------------------


def test_row_d_ambiguous_period_forces_needs_clarification():
    readings = [("FY2023 Q2", AnswerStatus.ANSWER, 100.0), ("FY2024 Q2", AnswerStatus.ANSWER, 200.0)]

    result = apply_gate(draft_status=AnswerStatus.ANSWER, period_readings=readings)

    assert result.final_status == AnswerStatus.NEEDS_CLARIFICATION
    assert set(result.clarification_options) == {"FY2023 Q2", "FY2024 Q2"}


def test_row_d_readings_within_tolerance_do_not_trigger_clarification():
    readings = [("FY2023 Q2", AnswerStatus.ANSWER, 100.0), ("FY2024 Q2", AnswerStatus.ANSWER, 101.0)]

    result = apply_gate(draft_status=AnswerStatus.ANSWER, period_readings=readings)

    assert result.final_status == AnswerStatus.ANSWER


def test_row_d_status_divergence_is_material_regardless_of_magnitude():
    readings = [("FY2023 Q2", AnswerStatus.ANSWER, 100.0), ("FY2024 Q2", AnswerStatus.REFUSED, 100.0)]

    assert readings_are_material_difference(readings) is True

    result = apply_gate(draft_status=AnswerStatus.ANSWER, period_readings=readings)
    assert result.final_status == AnswerStatus.NEEDS_CLARIFICATION


def test_materiality_function_zero_reference_guard():
    readings = [("A", AnswerStatus.ANSWER, 0.0), ("B", AnswerStatus.ANSWER, 0.0)]

    assert readings_are_material_difference(readings) is False


# ---------------------------------------------------------------------------
# Row E — non-authoritative grouping decision would change the result.
# ---------------------------------------------------------------------------


def test_row_e_grouping_decision_caps_at_partial():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, grouping_would_change_result=True)

    assert result.final_status == AnswerStatus.PARTIAL


def test_row_e_no_grouping_effect_does_not_degrade():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, grouping_would_change_result=False)

    assert result.final_status == AnswerStatus.ANSWER


# ---------------------------------------------------------------------------
# Row F — no data for the requested period.
# ---------------------------------------------------------------------------


def test_row_f_no_data_for_period_forces_refused():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=True)

    assert result.final_status == AnswerStatus.REFUSED
    assert result.refusal_reason is not None


# ---------------------------------------------------------------------------
# Precedence and combination.
# ---------------------------------------------------------------------------


def test_precedence_refused_dominates_needs_clarification():
    readings = [("FY2023 Q2", AnswerStatus.ANSWER, 100.0), ("FY2024 Q2", AnswerStatus.ANSWER, 500.0)]

    result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=True, period_readings=readings)

    assert result.final_status == AnswerStatus.REFUSED
    assert result.clarification_options == []
    assert "ambiguous_period" in result.firing_conditions


def test_precedence_refused_dominates_partial_ceiling():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=True, row_coverage=_PARTIAL_COVERAGE)

    assert result.final_status == AnswerStatus.REFUSED


def test_multiple_refusal_reasons_are_concatenated_not_dropped():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, missing_denominator=True, no_data_for_period=True)

    assert result.final_status == AnswerStatus.REFUSED
    assert "denominator" in result.refusal_reason
    assert "period" in result.refusal_reason


def test_warnings_added_include_all_fired_conditions_even_when_dominated():
    result = apply_gate(draft_status=AnswerStatus.ANSWER, no_data_for_period=True, row_coverage=_PARTIAL_COVERAGE)

    assert result.final_status == AnswerStatus.REFUSED
    assert any("coverage" in w for w in result.warnings_added)
    assert any("period" in w for w in result.warnings_added)
