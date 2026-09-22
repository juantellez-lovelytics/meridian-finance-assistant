"""Tests for finance_assistant.tools.te_policy: evaluate_te_policy (five-state
rule engine over config/policy_rules.yaml) and the two-basis T&E perimeter
derivation.

Concrete values live only in synthetic tmp_path fixtures built via the
`write_csv` fixture + the real loaders, never against the real data/*.csv
files (except test_load_policy_rules_round_trips_real_config, which is
intentionally checking the checked-in config/policy_rules.yaml's own content
-- the YAML *is* the thing under test there). Expected observed/threshold
values are hand-computed, never derived by calling the function under test.
"""

import dataclasses

import pytest

from finance_assistant import config
from finance_assistant.data.loaders import (
    load_chart_of_accounts,
    load_fx_rates,
    load_gl_transactions,
)
from finance_assistant.tools.te_policy import (
    AirfareRule,
    ClientEntertainmentRule,
    LodgingRule,
    NonReimbursableRule,
    PerDiemRule,
    PolicyFindingState,
    PolicyRules,
    PreApprovalRule,
    TEPerimeterRule,
    evaluate_te_policy,
    load_policy_rules,
)

# ---------------------------------------------------------------------------
# Fixture factories.
# ---------------------------------------------------------------------------


def _gl_row(**overrides):
    row = {
        "txn_id": "T0001",
        "posting_date": "2024-03-01",
        "accrual_date": "2024-03-01",
        "entity": "MI-US",
        "cost_centre": "OPS-NA",
        "account_code": "6220",
        "amount": "100.00",
        "currency": "USD",
        "vendor_id": "V1001",
        "doc_ref": "INV-0001",
        "approval_ref": "",
        "memo": "test row",
    }
    row.update(overrides)
    return row


def _coa_row(**overrides):
    row = {
        "account_code": "6220",
        "account_name": "Hotels & Lodging",
        "parent_code": "6200",
        "parent_name": "Travel & Entertainment",
        "statement_line": "Operating Expenses",
        "valid_from": "2023-01-01",
        "valid_to": "9999-12-31",
    }
    row.update(overrides)
    return row


def _fx_row(**overrides):
    row = {"period_month": "2024-03", "currency": "USD", "rate_to_usd": "1.0"}
    row.update(overrides)
    return row


def _always_te_coa():
    return [
        _coa_row(account_code="6210", account_name="Airfare"),
        _coa_row(account_code="6220", account_name="Hotels & Lodging"),
        _coa_row(account_code="6230", account_name="Meals & Entertainment"),
    ]


def _policy(**overrides):
    base = PolicyRules(
        source_document="travel_expense_policy.md",
        te_perimeter=TEPerimeterRule(parent_name="Travel & Entertainment", source_section="Scope"),
        lodging=LodgingRule(
            account_name="Hotels & Lodging",
            tier_one_cap_usd=275.0,
            standard_cap_usd=190.0,
            tier_one_cities=["New York", "London"],
            source_section="Lodging",
        ),
        airfare=AirfareRule(account_name="Airfare", business_class_keyword="business", source_section="Air travel"),
        per_diem=PerDiemRule(
            account_name="Meals & Entertainment",
            team_meals_keyword="team meals",
            daily_cap_usd=85.0,
            source_section="Meals",
        ),
        client_entertainment=ClientEntertainmentRule(
            account_name="Meals & Entertainment",
            keyword="client entertainment",
            approval_threshold_usd=500.0,
            source_section="Client entertainment",
        ),
        pre_approval=PreApprovalRule(threshold_usd=1000.0, source_section="Pre-approval"),
        non_reimbursable=NonReimbursableRule(
            keywords=["airline lounge", "minibar"], source_section="Non-reimbursable"
        ),
    )
    return dataclasses.replace(base, **overrides)


def _load_gl(write_csv, rows):
    path = write_csv(config.GL_SCHEMA["filename"], config.GL_SCHEMA["required_columns"], rows)
    return load_gl_transactions(path)


def _load_coa(write_csv, rows):
    path = write_csv(config.COA_SCHEMA["filename"], config.COA_SCHEMA["required_columns"], rows)
    return load_chart_of_accounts(path)


def _load_fx(write_csv, rows):
    path = write_csv(config.FX_SCHEMA["filename"], config.FX_SCHEMA["required_columns"], rows)
    return load_fx_rates(path)


def _evaluate(write_csv, gl_rows, fx_rows, coa_rows=None, policy=None, perimeter_basis="reported"):
    gl = _load_gl(write_csv, gl_rows)
    coa = _load_coa(write_csv, coa_rows if coa_rows is not None else _always_te_coa())
    fx = _load_fx(write_csv, fx_rows)
    return evaluate_te_policy(gl, coa, fx, policy or _policy(), perimeter_basis=perimeter_basis)


def _findings_by_rule(result, txn_id):
    return {f.rule_id: f for f in result.findings if f.txn_id == txn_id}


# ---------------------------------------------------------------------------
# load_policy_rules.
# ---------------------------------------------------------------------------


def test_load_policy_rules_round_trips_real_config():
    policy = load_policy_rules()

    assert policy.source_document == "travel_expense_policy.md"
    assert policy.te_perimeter.parent_name == "Travel & Entertainment"
    assert policy.lodging.tier_one_cap_usd == pytest.approx(275.0)
    assert policy.lodging.standard_cap_usd == pytest.approx(190.0)
    assert "New York" in policy.lodging.tier_one_cities
    assert policy.per_diem.daily_cap_usd == pytest.approx(85.0)
    assert policy.client_entertainment.approval_threshold_usd == pytest.approx(500.0)
    assert policy.pre_approval.threshold_usd == pytest.approx(1000.0)
    assert "minibar" in policy.non_reimbursable.keywords


def test_load_policy_rules_missing_section_raises(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("source_document: x\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_policy_rules(bad_yaml)


# ---------------------------------------------------------------------------
# Perimeter: two explicit, traceable bases (the correction applied after
# initial plan review).
# ---------------------------------------------------------------------------


def _mid_year_reparent_coa():
    return [
        _coa_row(
            account_code="6230",
            account_name="Meals & Entertainment",
            parent_code="6200",
            parent_name="Travel & Entertainment",
            valid_from="2023-01-01",
            valid_to="2024-06-30",
        ),
        _coa_row(
            account_code="6230",
            account_name="Meals & Entertainment",
            parent_code="6700",
            parent_name="Marketing",
            valid_from="2024-07-01",
            valid_to="9999-12-31",
        ),
    ]


def _mid_year_reparent_gl():
    return [
        _gl_row(
            txn_id="T_BEFORE",
            account_code="6230",
            accrual_date="2024-03-01",
            posting_date="2024-03-01",
            memo="client entertainment London",
            amount="600.00",
            approval_ref="AP-1",
        ),
        _gl_row(
            txn_id="T_AFTER",
            account_code="6230",
            accrual_date="2024-08-01",
            posting_date="2024-08-01",
            memo="client entertainment London",
            amount="600.00",
            approval_ref="AP-2",
        ),
    ]


def test_perimeter_reported_basis_excludes_post_reclassification_row(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=_mid_year_reparent_gl(),
        fx_rows=[_fx_row(period_month="2024-03"), _fx_row(period_month="2024-08")],
        coa_rows=_mid_year_reparent_coa(),
        perimeter_basis="reported",
    )

    assert result.perimeter_rows == 1
    assert result.rows_excluded_by_reclassification.row_count == 1
    assert result.rows_excluded_by_reclassification.account_codes == ["6230"]
    assert result.warnings
    assert "T_AFTER" not in {f.txn_id for f in result.findings}


def test_perimeter_policy_basis_includes_post_reclassification_row(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=_mid_year_reparent_gl(),
        fx_rows=[_fx_row(period_month="2024-03"), _fx_row(period_month="2024-08")],
        coa_rows=_mid_year_reparent_coa(),
        perimeter_basis="policy",
    )

    assert result.perimeter_rows == 2
    assert result.warnings == []
    # Diagnostic is always computed, regardless of which basis produced findings.
    assert result.rows_excluded_by_reclassification.row_count == 1

    after = _findings_by_rule(result, "T_AFTER")
    assert after["client_entertainment_approval"].state == PolicyFindingState.NOT_A_BREACH


def test_invalid_perimeter_basis_raises(write_csv):
    with pytest.raises(ValueError):
        _evaluate(write_csv, gl_rows=[_gl_row()], fx_rows=[_fx_row()], perimeter_basis="bogus")


# ---------------------------------------------------------------------------
# lodging_nightly_cap.
# ---------------------------------------------------------------------------


def test_lodging_potential_breach_above_tier_one_cap(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel - 2 nights, London", amount="600.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["lodging_nightly_cap"]
    assert finding.state == PolicyFindingState.POTENTIAL_BREACH
    assert finding.observed_value == pytest.approx(300.0)
    assert finding.threshold == pytest.approx(275.0)


def test_lodging_not_a_breach_within_cap(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel - 2 nights, London", amount="500.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["lodging_nightly_cap"]
    assert finding.state == PolicyFindingState.NOT_A_BREACH
    assert finding.observed_value == pytest.approx(250.0)


def test_lodging_insufficient_evidence_when_memo_unparseable(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel stay, unspecified")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["lodging_nightly_cap"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


def test_lodging_insufficient_evidence_when_fx_missing(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(txn_id="T1", account_code="6220", memo="hotel - 2 nights, London", currency="EUR")
        ],
        fx_rows=[_fx_row(currency="USD")],  # no EUR rate for the row's period_month
    )
    finding = _findings_by_rule(result, "T1")["lodging_nightly_cap"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# business_class_approval.
# ---------------------------------------------------------------------------


def test_business_class_confirmed_without_approval(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(txn_id="T1", account_code="6210", memo="airfare Zurich, business", amount="2000.00", approval_ref="")
        ],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["business_class_approval"]
    assert finding.state == PolicyFindingState.CONFIRMED_RULE_MATCH


def test_business_class_insufficient_evidence_when_approved(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(
                txn_id="T1", account_code="6210", memo="airfare Zurich, business", amount="2000.00", approval_ref="AP-1"
            )
        ],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["business_class_approval"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


def test_business_class_not_applicable_for_economy(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6210", memo="airfare Zurich, economy", amount="500.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["business_class_approval"]
    assert finding.state == PolicyFindingState.NOT_APPLICABLE


def test_business_class_insufficient_evidence_when_class_unparseable(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6210", memo="airfare Zurich", amount="500.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["business_class_approval"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# per_diem_daily_cap.
# ---------------------------------------------------------------------------


def test_per_diem_potential_breach_above_cap(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6230", memo="team meals 2 days, Zurich", amount="300.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["per_diem_daily_cap"]
    assert finding.state == PolicyFindingState.POTENTIAL_BREACH
    assert finding.observed_value == pytest.approx(150.0)
    assert finding.threshold == pytest.approx(85.0)


def test_per_diem_not_a_breach_within_cap(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6230", memo="team meals 2 days, Zurich", amount="100.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["per_diem_daily_cap"]
    assert finding.state == PolicyFindingState.NOT_A_BREACH


def test_per_diem_insufficient_evidence_without_day_count(write_csv):
    # This is the real dataset's actual memo shape -- no day-count is ever
    # present, so INSUFFICIENT_EVIDENCE is the correct generic outcome here,
    # not a coded special case.
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6230", memo="team meals Zurich", amount="100.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["per_diem_daily_cap"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


def test_per_diem_not_applicable_for_client_entertainment_memo(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6230", memo="client entertainment Zurich", amount="600.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["per_diem_daily_cap"]
    assert finding.state == PolicyFindingState.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# client_entertainment_approval / client_entertainment_attendee_record.
# ---------------------------------------------------------------------------


def test_client_entertainment_emits_two_independent_findings(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(
                txn_id="T1", account_code="6230", memo="client entertainment Zurich", amount="600.00", approval_ref=""
            )
        ],
        fx_rows=[_fx_row()],
    )
    findings = _findings_by_rule(result, "T1")

    assert findings["client_entertainment_approval"].state == PolicyFindingState.CONFIRMED_RULE_MATCH
    assert findings["client_entertainment_attendee_record"].state == PolicyFindingState.INSUFFICIENT_EVIDENCE
    assert findings["client_entertainment_approval"].rule_id != findings["client_entertainment_attendee_record"].rule_id


def test_client_entertainment_not_a_breach_when_approved(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(
                txn_id="T1",
                account_code="6230",
                memo="client entertainment Zurich",
                amount="600.00",
                approval_ref="AP-1",
            )
        ],
        fx_rows=[_fx_row()],
    )
    findings = _findings_by_rule(result, "T1")

    assert findings["client_entertainment_approval"].state == PolicyFindingState.NOT_A_BREACH
    # Approval doesn't change the attendee sub-rule's evaluability.
    assert findings["client_entertainment_attendee_record"].state == PolicyFindingState.INSUFFICIENT_EVIDENCE


def test_client_entertainment_attendee_record_has_no_threshold_gate(write_csv):
    # Regression guard: the source document's "Client entertainment" section
    # is two sentences -- only the VP sign-off clause carries the USD 500
    # figure ("...over USD 500 per event requires the sign-off of a Vice
    # President..."); the attendee-roster clause ("The names and employers of
    # all attendees must be recorded") has no threshold and applies to every
    # client-entertainment row in the perimeter. A below-threshold row must
    # therefore split: NOT_APPLICABLE on approval, INSUFFICIENT_EVIDENCE on
    # attendee record -- never NOT_APPLICABLE on both.
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6230", memo="client entertainment Zurich", amount="400.00")],
        fx_rows=[_fx_row()],
    )
    findings = _findings_by_rule(result, "T1")

    assert findings["client_entertainment_approval"].state == PolicyFindingState.NOT_APPLICABLE
    assert findings["client_entertainment_attendee_record"].state == PolicyFindingState.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# pre_approval_threshold.
# ---------------------------------------------------------------------------


def test_pre_approval_confirmed_without_reference(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel - 5 nights, London", amount="1500.00", approval_ref="")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["pre_approval_threshold"]
    assert finding.state == PolicyFindingState.CONFIRMED_RULE_MATCH
    assert finding.threshold == pytest.approx(1000.0)


def test_pre_approval_not_a_breach_with_reference(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(
                txn_id="T1", account_code="6220", memo="hotel - 5 nights, London", amount="1500.00", approval_ref="AP-1"
            )
        ],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["pre_approval_threshold"]
    assert finding.state == PolicyFindingState.NOT_A_BREACH


def test_pre_approval_not_applicable_below_threshold(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel - 1 nights, London", amount="500.00")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["pre_approval_threshold"]
    assert finding.state == PolicyFindingState.NOT_APPLICABLE


def test_pre_approval_insufficient_evidence_when_fx_missing(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6220", memo="hotel - 5 nights, London", amount="1500.00", currency="EUR")],
        fx_rows=[_fx_row(currency="USD")],
    )
    finding = _findings_by_rule(result, "T1")["pre_approval_threshold"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# non_reimbursable_keyword.
# ---------------------------------------------------------------------------


def test_non_reimbursable_confirmed_on_keyword_match(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6210", memo="airline lounge membership renewal")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["non_reimbursable_keyword"]
    assert finding.state == PolicyFindingState.CONFIRMED_RULE_MATCH
    assert finding.threshold == "airline lounge"


def test_non_reimbursable_not_a_breach_without_keyword(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6210", memo="airfare Zurich, economy")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["non_reimbursable_keyword"]
    assert finding.state == PolicyFindingState.NOT_A_BREACH


def test_non_reimbursable_insufficient_evidence_when_memo_blank(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[_gl_row(txn_id="T1", account_code="6210", memo="")],
        fx_rows=[_fx_row()],
    )
    finding = _findings_by_rule(result, "T1")["non_reimbursable_keyword"]
    assert finding.state == PolicyFindingState.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Reversal/credit rows: NOT_APPLICABLE across every rule.
# ---------------------------------------------------------------------------


def test_non_positive_amount_is_not_applicable_across_all_rules(write_csv):
    result = _evaluate(
        write_csv,
        gl_rows=[
            _gl_row(
                txn_id="T1",
                account_code="6220",
                memo="credit memo - reversal of INV-0001",
                amount="-500.00",
                doc_ref="CM-0001",
            )
        ],
        fx_rows=[_fx_row()],
    )
    findings = _findings_by_rule(result, "T1")

    assert len(findings) == 7
    assert all(f.state == PolicyFindingState.NOT_APPLICABLE for f in findings.values())


def test_missing_required_column_raises(write_csv):
    gl = _load_gl(write_csv, [_gl_row()]).drop(columns=["memo"])
    coa = _load_coa(write_csv, _always_te_coa())
    fx = _load_fx(write_csv, [_fx_row()])

    with pytest.raises(ValueError):
        evaluate_te_policy(gl, coa, fx, _policy())


# ---------------------------------------------------------------------------
# Structural smoke tests against real data/ -- no concrete values/counts.
# ---------------------------------------------------------------------------

_KNOWN_RULE_IDS = {
    "lodging_nightly_cap",
    "business_class_approval",
    "per_diem_daily_cap",
    "client_entertainment_approval",
    "client_entertainment_attendee_record",
    "pre_approval_threshold",
    "non_reimbursable_keyword",
}


def test_evaluate_te_policy_structural_smoke_against_real_data():
    gl = load_gl_transactions()
    coa = load_chart_of_accounts()
    fx = load_fx_rates()
    policy = load_policy_rules()

    reported = evaluate_te_policy(gl, coa, fx, policy, perimeter_basis="reported")
    policy_basis = evaluate_te_policy(gl, coa, fx, policy, perimeter_basis="policy")

    for result in (reported, policy_basis):
        assert result.limitation
        assert result.perimeter_rows + result.unmapped_account_rows <= result.total_rows
        assert {f.rule_id for f in result.findings} <= _KNOWN_RULE_IDS
        assert all(f.source_section for f in result.findings)
        # Every perimeter row gets exactly one finding per known rule.
        assert len(result.findings) == len(_KNOWN_RULE_IDS) * result.perimeter_rows

    # policy basis is always a superset of reported basis.
    assert policy_basis.perimeter_rows >= reported.perimeter_rows

    # The real dataset does have a mid-year reclassification (account 6230),
    # so this exclusion is expected to be nonzero -- asserted as > 0, never
    # against a specific count.
    assert reported.rows_excluded_by_reclassification.row_count > 0
    assert reported.warnings
    assert policy_basis.rows_excluded_by_reclassification.row_count == reported.rows_excluded_by_reclassification.row_count
    assert policy_basis.warnings == []
