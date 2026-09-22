"""T&E policy rule engine, reading thresholds from config/policy_rules.yaml.

Rules are never embedded in code (see `load_policy_rules`): every threshold,
city list, and keyword lives in the YAML file and cites the section of
data/documents/travel_expense_policy.md it came from, so a finding can quote
its exact source.

Perimeter (see `_resolve_te_perimeter`): the T&E policy applies to whatever
the chart of accounts calls "Travel & Entertainment", derived from the
*temporal* COA join (R3), never a static account-code list -- a mid-year
reclassification moves which account_codes currently report under that
parent. Two named bases are exposed rather than one hardcoded interpretation:
"reported" (current-period COA parent, what external reporting sees) and
"policy" (every account_code that was ever T&E anywhere in the COA's date
range). A reporting-line reclassification does not amend the policy document,
so rows excluded under "reported" are always counted and surfaced, never
silently dropped.

Every finding is a five-state candidate (CONFIRMED_RULE_MATCH,
POTENTIAL_BREACH, INSUFFICIENT_EVIDENCE, NOT_A_BREACH, NOT_APPLICABLE), never
a verdict: this module is a detector, not an auditor. Amounts absent from
this dataset (flight duration, per-diem day-counts, attendee rosters) are
never inferred -- most visibly, flight duration is never inferred from the
destination city.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml

from finance_assistant import config
from finance_assistant.config import DEFAULT_FINANCIAL_DATE_FIELD
from finance_assistant.tools.accounts import resolve_account_hierarchy
from finance_assistant.tools.fx import convert_to_usd

# ---------------------------------------------------------------------------
# Policy config -- parsed from config/policy_rules.yaml, never hardcoded here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TEPerimeterRule:
    parent_name: str
    source_section: str


@dataclass(frozen=True)
class LodgingRule:
    account_name: str
    tier_one_cap_usd: float
    standard_cap_usd: float
    tier_one_cities: list[str]
    source_section: str


@dataclass(frozen=True)
class AirfareRule:
    account_name: str
    business_class_keyword: str
    source_section: str


@dataclass(frozen=True)
class PerDiemRule:
    account_name: str
    team_meals_keyword: str
    daily_cap_usd: float
    source_section: str


@dataclass(frozen=True)
class ClientEntertainmentRule:
    account_name: str
    keyword: str
    approval_threshold_usd: float
    source_section: str


@dataclass(frozen=True)
class PreApprovalRule:
    threshold_usd: float
    source_section: str


@dataclass(frozen=True)
class NonReimbursableRule:
    keywords: list[str]
    source_section: str


@dataclass(frozen=True)
class PolicyRules:
    source_document: str
    te_perimeter: TEPerimeterRule
    lodging: LodgingRule
    airfare: AirfareRule
    per_diem: PerDiemRule
    client_entertainment: ClientEntertainmentRule
    pre_approval: PreApprovalRule
    non_reimbursable: NonReimbursableRule


_REQUIRED_POLICY_SECTIONS = [
    "source_document",
    "te_perimeter",
    "lodging",
    "airfare",
    "per_diem",
    "client_entertainment",
    "pre_approval",
    "non_reimbursable",
]


def load_policy_rules(path: str | Path | None = None) -> PolicyRules:
    resolved = Path(path) if path is not None else config.CONFIG_DIR / config.POLICY_RULES_FILE
    with open(resolved, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    missing_sections = [s for s in _REQUIRED_POLICY_SECTIONS if s not in raw]
    if missing_sections:
        raise ValueError(f"{resolved} is missing required section(s): {', '.join(missing_sections)}")

    try:
        return PolicyRules(
            source_document=raw["source_document"],
            te_perimeter=TEPerimeterRule(**raw["te_perimeter"]),
            lodging=LodgingRule(**raw["lodging"]),
            airfare=AirfareRule(**raw["airfare"]),
            per_diem=PerDiemRule(**raw["per_diem"]),
            client_entertainment=ClientEntertainmentRule(**raw["client_entertainment"]),
            pre_approval=PreApprovalRule(**raw["pre_approval"]),
            non_reimbursable=NonReimbursableRule(**raw["non_reimbursable"]),
        )
    except TypeError as exc:
        raise ValueError(f"{resolved} has a malformed rule section: {exc}") from exc


# ---------------------------------------------------------------------------
# Perimeter derivation -- two explicit, traceable bases.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReclassificationExclusion:
    row_count: int
    account_codes: list[str]


def _policy_perimeter_account_codes(coa: pd.DataFrame, parent_name: str) -> set[str]:
    """account_codes that had this parent_name at ANY point in the COA's date
    range -- discovered generically from the COA, no account_code hardcoded."""
    return set(coa.loc[coa["parent_name"] == parent_name, "account_code"].unique())


def _resolve_te_perimeter(
    rows: pd.DataFrame,
    coa: pd.DataFrame,
    policy: PolicyRules,
    perimeter_basis: str,
    date_field: str,
):
    if perimeter_basis not in ("reported", "policy"):
        raise ValueError(f"perimeter_basis must be 'reported' or 'policy', got {perimeter_basis!r}")

    hierarchy = resolve_account_hierarchy(rows, coa, date_field=date_field, strict=False)
    reported_mask = hierarchy.rows["parent_name"] == policy.te_perimeter.parent_name

    policy_codes = _policy_perimeter_account_codes(coa, policy.te_perimeter.parent_name)
    policy_mask = hierarchy.rows["account_code"].isin(policy_codes)

    excluded_mask = policy_mask & ~reported_mask
    excluded = ReclassificationExclusion(
        row_count=int(excluded_mask.sum()),
        account_codes=sorted(hierarchy.rows.loc[excluded_mask, "account_code"].unique().tolist()),
    )

    selected_mask = reported_mask if perimeter_basis == "reported" else policy_mask
    return hierarchy.rows.loc[selected_mask].copy(), hierarchy, excluded


# ---------------------------------------------------------------------------
# Findings.
# ---------------------------------------------------------------------------


class PolicyFindingState(str, Enum):
    CONFIRMED_RULE_MATCH = "CONFIRMED_RULE_MATCH"
    POTENTIAL_BREACH = "POTENTIAL_BREACH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_A_BREACH = "NOT_A_BREACH"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class PolicyFinding:
    txn_id: str
    rule_id: str
    state: PolicyFindingState
    observed_value: object
    threshold: object
    reason: str
    source_document: str
    source_section: str


TE_POLICY_LIMITATION = (
    "Findings are rule-match candidates for review, never a definitive compliance "
    "audit. Business-class flight duration and per-diem day-counts are absent from "
    "this dataset and are never inferred from the destination city, so those rules "
    "surface as INSUFFICIENT_EVIDENCE rather than a guessed verdict."
)


@dataclass(frozen=True)
class TEPolicyEvaluationResult:
    findings: list[PolicyFinding]
    perimeter_basis: str  # "reported" | "policy" -- which basis produced `findings`
    perimeter_rows: int
    total_rows: int
    unmapped_account_rows: int
    rows_excluded_by_reclassification: ReclassificationExclusion
    warnings: list[str]
    limitation: str = TE_POLICY_LIMITATION


# ---------------------------------------------------------------------------
# Per-rule evaluation. Each function returns (state, observed_value,
# threshold, reason); rule_id / source_document / source_section are attached
# by the caller.
# ---------------------------------------------------------------------------

_LODGING_MEMO_PATTERN = re.compile(r"(\d+)\s*nights?,\s*(.+)$", re.IGNORECASE)
_AIRFARE_CLASS_PATTERN = re.compile(r"\b(economy|business)\b", re.IGNORECASE)
_DAY_COUNT_PATTERN = re.compile(r"(\d+)\s*days?\b", re.IGNORECASE)


def _account_matches(record: dict, expected_account_name: str) -> bool | None:
    """True/False if the row's resolved account_name is known; None if the
    account_code itself could not be resolved (unmapped in the temporal COA)."""
    account_name = record.get("account_name")
    if pd.isna(account_name):
        return None
    return account_name == expected_account_name


def _blank(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def _eval_lodging(record: dict, policy: PolicyRules):
    rule = policy.lodging
    match = _account_matches(record, rule.account_name)
    if match is None:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            "account category could not be resolved for this row (unmapped account_code in the chart of accounts)",
        )
    if not match:
        return PolicyFindingState.NOT_APPLICABLE, None, None, "row is not in the Hotels & Lodging category"
    if not record["is_fx_convertible"]:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            f"{rule.tier_one_cap_usd}/{rule.standard_cap_usd}",
            "missing FX rate to convert amount to USD",
        )
    memo = record.get("memo")
    parsed = _LODGING_MEMO_PATTERN.search(memo) if isinstance(memo, str) else None
    if not parsed or int(parsed.group(1)) <= 0:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            memo,
            None,
            "nights/city not parseable from memo; nightly rate cannot be computed",
        )
    nights = int(parsed.group(1))
    city = parsed.group(2).strip()
    is_tier_one = city.lower() in {c.lower() for c in rule.tier_one_cities}
    cap = rule.tier_one_cap_usd if is_tier_one else rule.standard_cap_usd
    nightly_usd = record["amount_usd"] / nights
    if nightly_usd > cap:
        return (
            PolicyFindingState.POTENTIAL_BREACH,
            round(nightly_usd, 2),
            cap,
            f"nightly rate exceeds the {'tier-one' if is_tier_one else 'standard'} cap; the cap is "
            "exclusive of local tax and the GL does not separate tax from the room charge",
        )
    return PolicyFindingState.NOT_A_BREACH, round(nightly_usd, 2), cap, "nightly rate within the applicable cap"


def _eval_business_class(record: dict, policy: PolicyRules):
    rule = policy.airfare
    match = _account_matches(record, rule.account_name)
    if match is None:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.business_class_keyword,
            "account category could not be resolved for this row (unmapped account_code in the chart of accounts)",
        )
    if not match:
        return PolicyFindingState.NOT_APPLICABLE, None, rule.business_class_keyword, "row is not in the Airfare category"
    memo = record.get("memo")
    parsed = _AIRFARE_CLASS_PATTERN.search(memo) if isinstance(memo, str) else None
    if not parsed:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            memo,
            rule.business_class_keyword,
            "cabin class not parseable from memo",
        )
    cabin_class = parsed.group(1).lower()
    if cabin_class != rule.business_class_keyword.lower():
        return (
            PolicyFindingState.NOT_APPLICABLE,
            cabin_class,
            rule.business_class_keyword,
            "economy fare; the business-class approval rule does not apply",
        )
    if _blank(record.get("approval_ref")):
        return (
            PolicyFindingState.CONFIRMED_RULE_MATCH,
            cabin_class,
            rule.business_class_keyword,
            "business class booked without a recorded VP approval reference",
        )
    return (
        PolicyFindingState.INSUFFICIENT_EVIDENCE,
        cabin_class,
        rule.business_class_keyword,
        "business class approved, but flight duration is absent from the data and is never inferred "
        "from the destination city, so the policy's >6h trigger cannot be confirmed or ruled out",
    )


def _eval_per_diem(record: dict, policy: PolicyRules):
    rule = policy.per_diem
    match = _account_matches(record, rule.account_name)
    if match is None:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.daily_cap_usd,
            "account category could not be resolved for this row (unmapped account_code in the chart of accounts)",
        )
    memo = record.get("memo")
    memo_lower = memo.lower() if isinstance(memo, str) else ""
    is_client_entertainment = policy.client_entertainment.keyword.lower() in memo_lower
    if not match or is_client_entertainment:
        return PolicyFindingState.NOT_APPLICABLE, None, rule.daily_cap_usd, "row is not a per-diem meals expense"
    if not record["is_fx_convertible"]:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.daily_cap_usd,
            "missing FX rate to convert amount to USD",
        )
    parsed = _DAY_COUNT_PATTERN.search(memo) if isinstance(memo, str) else None
    if not parsed or int(parsed.group(1)) <= 0:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            memo,
            rule.daily_cap_usd,
            "day-count not parseable from memo; per-diem daily rate cannot be computed",
        )
    days = int(parsed.group(1))
    daily_usd = record["amount_usd"] / days
    if daily_usd > rule.daily_cap_usd:
        return (
            PolicyFindingState.POTENTIAL_BREACH,
            round(daily_usd, 2),
            rule.daily_cap_usd,
            "daily meal rate exceeds the per-diem cap",
        )
    return (
        PolicyFindingState.NOT_A_BREACH,
        round(daily_usd, 2),
        rule.daily_cap_usd,
        "daily meal rate within the per-diem cap",
    )


def _client_entertainment_scope(record: dict, policy: PolicyRules) -> bool | None:
    rule = policy.client_entertainment
    match = _account_matches(record, rule.account_name)
    if match is None:
        return None
    memo = record.get("memo")
    memo_lower = memo.lower() if isinstance(memo, str) else ""
    return bool(match and rule.keyword.lower() in memo_lower)


def _eval_client_entertainment_approval(record: dict, policy: PolicyRules):
    rule = policy.client_entertainment
    in_scope = _client_entertainment_scope(record, policy)
    if in_scope is None:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.approval_threshold_usd,
            "account category could not be resolved for this row (unmapped account_code in the chart of accounts)",
        )
    if not in_scope:
        return (
            PolicyFindingState.NOT_APPLICABLE,
            None,
            rule.approval_threshold_usd,
            "row is not a client-entertainment expense",
        )
    if not record["is_fx_convertible"]:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.approval_threshold_usd,
            "missing FX rate to convert amount to USD",
        )
    amount_usd = record["amount_usd"]
    if amount_usd < rule.approval_threshold_usd:
        return (
            PolicyFindingState.NOT_APPLICABLE,
            round(amount_usd, 2),
            rule.approval_threshold_usd,
            "below the client-entertainment approval threshold",
        )
    if _blank(record.get("approval_ref")):
        return (
            PolicyFindingState.CONFIRMED_RULE_MATCH,
            round(amount_usd, 2),
            rule.approval_threshold_usd,
            "client entertainment above threshold without a recorded VP approval reference",
        )
    return (
        PolicyFindingState.NOT_A_BREACH,
        round(amount_usd, 2),
        rule.approval_threshold_usd,
        "VP approval reference recorded",
    )


def _eval_client_entertainment_attendee(record: dict, policy: PolicyRules):
    # Unlike _eval_client_entertainment_approval, this sub-rule carries no
    # amount threshold: the policy document's "Client entertainment" section
    # has two separate sentences -- the USD 500 figure belongs only to the
    # first ("...over USD 500 per event requires the sign-off of a Vice
    # President in advance"). The second sentence ("The names and employers
    # of all attendees must be recorded") is unconditional. Gating this
    # sub-rule on the same threshold as the approval sub-rule would be
    # reading a qualifier into a clause that doesn't have one.
    in_scope = _client_entertainment_scope(record, policy)
    if in_scope is None:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            "account category could not be resolved for this row (unmapped account_code in the chart of accounts)",
        )
    if not in_scope:
        return PolicyFindingState.NOT_APPLICABLE, None, None, "row is not a client-entertainment expense"
    return (
        PolicyFindingState.INSUFFICIENT_EVIDENCE,
        None,
        None,
        "attendee name/employer roster is not captured anywhere in gl_transactions.csv, so this "
        "sub-rule can never be confirmed from ledger data alone",
    )


def _eval_pre_approval(record: dict, policy: PolicyRules):
    rule = policy.pre_approval
    if not record["is_fx_convertible"]:
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            rule.threshold_usd,
            "missing FX rate to convert amount to USD",
        )
    amount_usd = record["amount_usd"]
    if amount_usd < rule.threshold_usd:
        return (
            PolicyFindingState.NOT_APPLICABLE,
            round(amount_usd, 2),
            rule.threshold_usd,
            "below the pre-approval threshold",
        )
    if _blank(record.get("approval_ref")):
        return (
            PolicyFindingState.CONFIRMED_RULE_MATCH,
            round(amount_usd, 2),
            rule.threshold_usd,
            "expense at or above the pre-approval threshold without a recorded approval reference",
        )
    return (
        PolicyFindingState.NOT_A_BREACH,
        round(amount_usd, 2),
        rule.threshold_usd,
        "approval reference recorded",
    )


def _eval_non_reimbursable(record: dict, policy: PolicyRules):
    rule = policy.non_reimbursable
    memo = record.get("memo")
    if not isinstance(memo, str) or not memo.strip():
        return (
            PolicyFindingState.INSUFFICIENT_EVIDENCE,
            None,
            None,
            "memo is missing; non-reimbursable keywords cannot be checked",
        )
    memo_lower = memo.lower()
    matched = next((kw for kw in rule.keywords if kw.lower() in memo_lower), None)
    if matched:
        return (
            PolicyFindingState.CONFIRMED_RULE_MATCH,
            memo,
            matched,
            f"memo matches non-reimbursable keyword '{matched}'",
        )
    return PolicyFindingState.NOT_A_BREACH, memo, None, "no non-reimbursable keyword found in memo"


_RULES = [
    ("lodging_nightly_cap", _eval_lodging, lambda p: p.lodging.source_section),
    ("business_class_approval", _eval_business_class, lambda p: p.airfare.source_section),
    ("per_diem_daily_cap", _eval_per_diem, lambda p: p.per_diem.source_section),
    ("client_entertainment_approval", _eval_client_entertainment_approval, lambda p: p.client_entertainment.source_section),
    (
        "client_entertainment_attendee_record",
        _eval_client_entertainment_attendee,
        lambda p: p.client_entertainment.source_section,
    ),
    ("pre_approval_threshold", _eval_pre_approval, lambda p: p.pre_approval.source_section),
    ("non_reimbursable_keyword", _eval_non_reimbursable, lambda p: p.non_reimbursable.source_section),
]

_REQUIRED_COLUMNS = ["txn_id", "account_code", "amount", "currency", "memo", "approval_ref"]


def evaluate_te_policy(
    rows: pd.DataFrame,
    coa: pd.DataFrame,
    fx: pd.DataFrame,
    policy: PolicyRules,
    perimeter_basis: str = "reported",
    date_field: str = DEFAULT_FINANCIAL_DATE_FIELD,
) -> TEPolicyEvaluationResult:
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in rows.columns]
    if missing_columns:
        raise ValueError(f"rows is missing required column(s): {', '.join(missing_columns)}")

    perimeter_rows, hierarchy, excluded = _resolve_te_perimeter(rows, coa, policy, perimeter_basis, date_field)
    fx_result = convert_to_usd(perimeter_rows, fx, date_field=date_field)
    evaluable = fx_result.rows

    findings: list[PolicyFinding] = []
    for record in evaluable.to_dict("records"):
        is_expense = record["amount"] > 0
        for rule_id, rule_fn, section_fn in _RULES:
            if not is_expense:
                state = PolicyFindingState.NOT_APPLICABLE
                observed, threshold = record["amount"], None
                reason = "reversal/credit entry, not a policy-evaluable expense"
            else:
                state, observed, threshold, reason = rule_fn(record, policy)
            findings.append(
                PolicyFinding(
                    txn_id=record["txn_id"],
                    rule_id=rule_id,
                    state=state,
                    observed_value=observed,
                    threshold=threshold,
                    reason=reason,
                    source_document=policy.source_document,
                    source_section=section_fn(policy),
                )
            )

    warnings: list[str] = []
    if perimeter_basis == "reported" and excluded.row_count > 0:
        warnings.append(
            f"{excluded.row_count} row(s) under account_code(s) {excluded.account_codes} were T&E "
            "policy scope at some point in this dataset's date range but fall outside the 'reported' "
            "perimeter basis due to a chart-of-accounts reclassification -- the T&E policy document "
            "was not amended by that reclassification. Re-run with perimeter_basis='policy' to include them."
        )

    return TEPolicyEvaluationResult(
        findings=findings,
        perimeter_basis=perimeter_basis,
        perimeter_rows=len(perimeter_rows),
        total_rows=len(rows),
        unmapped_account_rows=len(hierarchy.unmapped),
        rows_excluded_by_reclassification=excluded,
        warnings=warnings,
    )
