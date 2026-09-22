"""Tests for finance_assistant.orchestration.intents (Intent, IntentRequest)."""

import pytest
from pydantic import ValidationError

from finance_assistant.orchestration.intents import Intent, IntentRequest


def test_intent_request_accepts_valid_confidence():
    request = IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.9, quarter="Q2")
    assert request.confidence == 0.9
    assert request.quarter == "Q2"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_intent_request_rejects_confidence_outside_unit_range(confidence):
    with pytest.raises(ValidationError):
        IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=confidence)


def test_intent_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        IntentRequest(intent=Intent.OPEX_BY_COST_CENTRE, confidence=0.9, made_up_field="x")


def test_intent_request_defaults_all_params_to_none():
    request = IntentRequest(intent=Intent.UNKNOWN, confidence=0.0)
    assert request.quarter is None
    assert request.year is None
    assert request.year_current is None
    assert request.year_prior is None
    assert request.date_start is None
    assert request.date_end is None
    assert request.top_n is None
    assert request.perimeter_basis is None


def test_evidence_models_uses_the_orchestration_intent_not_a_duplicate():
    """Regression proof of the Fase H move: Intent used to live defined
    inside evidence.models (with a comment saying it was temporary); it now
    must be defined exactly once, in orchestration.intents, and
    evidence.models.EvidenceBundle.intent must be typed with that same
    object (imported, not redefined) -- `evidence.models` necessarily
    exposes the name as a module attribute because EvidenceBundle needs it
    as a real (non-TYPE_CHECKING) annotation, so the regression this test
    actually guards against is a second, divergent Intent class, not the
    name being importable from both places."""
    import finance_assistant.evidence.models as evidence_models
    from finance_assistant.evidence.models import EvidenceBundle

    assert evidence_models.Intent is Intent
    assert EvidenceBundle.model_fields["intent"].annotation is Intent
