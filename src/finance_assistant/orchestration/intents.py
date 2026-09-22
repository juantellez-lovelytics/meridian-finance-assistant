"""The intent taxonomy and the Question Interpreter's structured output.

`Intent` is consumed by `evidence.models.EvidenceBundle.intent` and by
every plan/registry lookup in `orchestration.plans`. It lives here, not in
`evidence/`, because the taxonomy belongs to the orchestration layer that
classifies questions into it -- `evidence/` only needs the type, and
importing it from here (rather than the other way around) keeps the
dependency one-directional: `orchestration -> evidence`, never the
reverse, so `orchestration.orchestrator` can freely import
`EvidenceBundle` from `evidence.models` without a cycle.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    OPEX_BY_COST_CENTRE = "opex_by_cost_centre"  # Q1 -> workflows/opex.py
    TRAVEL_COMPARISON = "travel_comparison"  # Q2 -> workflows/travel.py
    CONSOLIDATED_SPEND = "consolidated_spend"  # Q3 -> workflows/consolidated.py
    TOP_VENDORS = "top_vendors"  # Q4 -> workflows/vendors.py
    BUDGET_VARIANCE = "budget_variance"  # Q5 -> workflows/variance.py
    TE_POLICY_CHECK = "te_policy_check"  # Q6 -> workflows/policy.py
    HEADCOUNT_COST_PER_FTE = "headcount_cost_per_fte"  # Q7 -> workflows/headcount.py
    DUPLICATE_PAYMENT_CHECK = "duplicate_payment_check"  # Q8 -> workflows/duplicates.py
    UNKNOWN = "unknown"


class IntentRequest(BaseModel):
    """Structured output of the Question Interpreter's single LLM call (or
    the deterministic keyword fallback). Deliberately FLAT -- the union of
    every workflow's optional scalar parameters -- rather than a
    discriminated union keyed by intent: structured-output APIs turn one
    pydantic model into one static JSON schema, and provider-specific
    oneOf/discriminator support is inconsistent for no benefit here, since
    no field name means a different thing for two different intents.

    Fields that must never be model-settable are intentionally absent:
    `date_field` (R1 -- accrual_date is the permanent default), and
    `documents_dir`/`policy_path`/duplicate-check `rules` (injected
    plumbing, never a caller-supplied parameter -- see
    `orchestration.plans.assemble_kwargs`).
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)

    quarter: Literal["Q1", "Q2", "Q3", "Q4"] | None = None
    year: int | None = None
    year_current: int | None = None
    year_prior: int | None = None
    date_start: str | None = None
    date_end: str | None = None
    top_n: int | None = None
    perimeter_basis: Literal["reported", "policy"] | None = None
