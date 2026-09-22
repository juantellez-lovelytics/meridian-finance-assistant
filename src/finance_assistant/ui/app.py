"""Streamlit UI -- a pure view over `orchestration.orchestrator.answer_question`.

Same entry point as the CLI's free-text mode (`cli.py:_run_free_text_mode`):
this module never imports `workflows/`, `tools/`, `evidence.gate`, or
`orchestration.interpreter` directly, and never touches a DataFrame. Every
value shown here already exists on the returned `EvidenceBundle`/`RunTrace`
-- this file only decides how to lay those values out.

Run with: streamlit run src/finance_assistant/ui/app.py
"""

import pandas as pd
import streamlit as st

from finance_assistant import config
from finance_assistant.evidence.models import AnswerStatus, EvidenceBundle
from finance_assistant.evidence.trace import RunTrace
from finance_assistant.orchestration.intents import Intent
from finance_assistant.orchestration.orchestrator import answer_question
from finance_assistant.orchestration.settings import SUPPORTED_CREDENTIAL_ENV_VARS, load_settings

EXAMPLE_QUESTIONS = [
    "What was our opex by cost centre in Q2?",
    "How does travel & entertainment spend in the most recent year compare to the prior year?",
    "What was our consolidated spend in Q3, in USD?",
    "Who are our top 10 vendors by spend?",
    "Which cost centres performed worst against budget in Q3, and what's driving it?",
    "Which transactions breach our travel & entertainment policy?",
    "What's our headcount cost per FTE?",
    "Have we paid any vendor twice for the same thing?",
]

# (text color, background color) -- flat fills, no gradients/shadows.
STATUS_COLORS: dict[AnswerStatus, tuple[str, str]] = {
    AnswerStatus.ANSWER: ("#0f5132", "#d1e7dd"),
    AnswerStatus.PARTIAL: ("#664d03", "#fff3cd"),
    AnswerStatus.REFUSED: ("#842029", "#f8d7da"),
    AnswerStatus.NEEDS_CLARIFICATION: ("#084298", "#cfe2ff"),
    AnswerStatus.ERROR: ("#41464b", "#e2e3e5"),
}

BOUNDARY_PRINCIPLE = (
    "The system is agentic at the interpretation boundary and deterministic "
    "at the financial-computation boundary."
)

# T&E policy review: the two states that answer "which transactions breach
# policy". INSUFFICIENT_EVIDENCE is real information but not itself a
# finding, so it's reported separately, never folded in with these.
TE_ACTIONABLE_STATES = ("CONFIRMED_RULE_MATCH", "POTENTIAL_BREACH")
# NOT_APPLICABLE/NOT_A_BREACH are coverage bookkeeping, not findings -- most
# rule x transaction pairs simply don't apply. NOT_APPLICABLE in particular
# dominates raw counts (rules x transactions, not breaches) and would swamp
# any chart or metric it shares with actual findings.
TE_COVERAGE_STATES = ("NOT_APPLICABLE", "NOT_A_BREACH")
TE_STATE_LABELS = {
    "CONFIRMED_RULE_MATCH": "Confirmed rule match",
    "POTENTIAL_BREACH": "Potential breach",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
    "NOT_A_BREACH": "Not a breach",
    "NOT_APPLICABLE": "Not applicable",
}


def _set_question(question: str) -> None:
    st.session_state["question"] = question


def render_status_badge(status: AnswerStatus) -> None:
    text_color, bg_color = STATUS_COLORS[status]
    st.markdown(
        f'<div style="background-color:{bg_color};color:{text_color};'
        'padding:0.6rem 1.1rem;border-radius:0.4rem;font-size:1.3rem;'
        'font-weight:700;display:inline-block;letter-spacing:0.03em;">'
        f"{status.value.upper().replace('_', ' ')}</div>",
        unsafe_allow_html=True,
    )


def render_coverage_block(bundle: EvidenceBundle) -> None:
    coverage = bundle.coverage
    fraction = min(max(coverage.computable_amount_pct / 100.0, 0.0), 1.0)
    st.progress(fraction)
    st.markdown(
        f"**{coverage.computable_rows:,} of {coverage.selected_rows:,} rows computable "
        f"({coverage.computable_amount_pct:.1f}%)**"
    )


def _basis_column(label: str, basis: dict, caption: str | None = None) -> None:
    st.markdown(f"**{label}**")
    st.metric("current", f"${basis['current']['te_total_usd']:,.2f}")
    st.metric("prior", f"${basis['prior']['te_total_usd']:,.2f}")
    st.metric("variance", f"${basis['variance_usd']:,.2f}")
    if caption:
        st.caption(caption)


def render_travel_comparison(result: dict) -> set[str]:
    """Two legitimate readings, side by side: comparable basis anchored to the
    most recent date in range vs. the earliest. Reported basis shown alongside
    as the (non-authoritative) naive baseline. See workflows/travel.py."""
    cols = st.columns(3)
    with cols[0]:
        _basis_column("Reported basis", result["reported_basis"])
    with cols[1]:
        _basis_column(
            "Comparable basis (ref: most recent)",
            result["comparable_basis_current"],
            f"reference date: {result['comparable_basis_current']['reference_date']}",
        )
    with cols[2]:
        _basis_column(
            "Comparable basis (ref: earliest)",
            result["comparable_basis_prior"],
            f"reference date: {result['comparable_basis_prior']['reference_date']}",
        )
    return {"reported_basis", "comparable_basis_current", "comparable_basis_prior"}


def render_top_vendors(result: dict) -> set[str]:
    """Two legitimate readings, side by side: authoritative vendor_id ranking
    vs. the candidate alias-cluster ranking. See workflows/vendors.py."""
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Ranking by vendor_id** (authoritative)")
        st.dataframe(pd.DataFrame(result["ranking_by_vendor_id"]), hide_index=True, use_container_width=True)
    with cols[1]:
        st.markdown("**Ranking by alias cluster** (candidate)")
        st.dataframe(pd.DataFrame(result["ranking_by_alias_cluster"]), hide_index=True, use_container_width=True)
    return {"ranking_by_vendor_id", "ranking_by_alias_cluster"}


def render_te_policy_review(result: dict) -> set[str]:
    """CONFIRMED_RULE_MATCH/POTENTIAL_BREACH answer "which transactions
    breach policy" and lead the view. INSUFFICIENT_EVIDENCE gets its own
    metric -- valuable ("what couldn't be checked"), but not a finding.
    NOT_APPLICABLE/NOT_A_BREACH are evaluation coverage (rules x
    transactions that didn't apply, or applied and cleared) and go in an
    expander so they never swamp the actionable numbers. See
    workflows/policy.py."""
    by_state: dict[str, int] = result.get("findings_by_state", {})
    by_rule: dict[str, dict[str, int]] = result.get("findings_by_rule", {})

    st.markdown("**Policy findings**")
    cols = st.columns(2)
    with cols[0]:
        st.metric(TE_STATE_LABELS["CONFIRMED_RULE_MATCH"], f"{by_state.get('CONFIRMED_RULE_MATCH', 0):,}")
    with cols[1]:
        st.metric(TE_STATE_LABELS["POTENTIAL_BREACH"], f"{by_state.get('POTENTIAL_BREACH', 0):,}")

    st.metric(
        "Could not be evaluated (insufficient evidence)",
        f"{by_state.get('INSUFFICIENT_EVIDENCE', 0):,}",
    )

    chart_data = {
        TE_STATE_LABELS[state]: by_state[state]
        for state in (*TE_ACTIONABLE_STATES, "INSUFFICIENT_EVIDENCE")
        if by_state.get(state, 0) > 0
    }
    if chart_data:
        st.bar_chart(pd.Series(chart_data, name="count"))

    rule_rows = [
        {
            "rule": rule_id.replace("_", " ").capitalize(),
            TE_STATE_LABELS["CONFIRMED_RULE_MATCH"]: states.get("CONFIRMED_RULE_MATCH", 0),
            TE_STATE_LABELS["POTENTIAL_BREACH"]: states.get("POTENTIAL_BREACH", 0),
        }
        for rule_id, states in sorted(by_rule.items())
        if states.get("CONFIRMED_RULE_MATCH", 0) or states.get("POTENTIAL_BREACH", 0)
    ]
    if rule_rows:
        st.markdown("**Confirmed findings by rule**")
        st.dataframe(pd.DataFrame(rule_rows), hide_index=True, use_container_width=True)

    with st.expander("Evaluation coverage (not findings)"):
        st.caption(
            "Total evaluations = rules x transactions, not a breach count -- "
            "most rule/transaction pairs simply don't apply (e.g. a lodging "
            "cap rule evaluated against a non-lodging expense)."
        )
        coverage_rows = [
            {"state": TE_STATE_LABELS[state], "count": by_state[state]}
            for state in TE_COVERAGE_STATES
            if state in by_state
        ]
        if coverage_rows:
            st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, use_container_width=True)

    return {"findings_by_state", "findings_by_rule"}


def render_result(result: dict, intent: Intent) -> None:
    shown: set[str] = set()
    if intent == Intent.TRAVEL_COMPARISON and "reported_basis" in result:
        shown = render_travel_comparison(result)
    elif intent == Intent.TOP_VENDORS and "ranking_by_vendor_id" in result:
        shown = render_top_vendors(result)
    elif intent == Intent.TE_POLICY_CHECK and "findings_by_state" in result:
        shown = render_te_policy_review(result)

    remaining = {k: v for k, v in result.items() if k not in shown}
    if not remaining:
        return

    list_fields = {
        k: v for k, v in remaining.items() if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)
    }
    numeric_dict_fields = {
        k: v
        for k, v in remaining.items()
        if isinstance(v, dict) and v and all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in v.values())
    }
    scalar_fields = {
        k: v for k, v in remaining.items() if k not in list_fields and k not in numeric_dict_fields and not isinstance(v, (dict, list))
    }
    other_fields = {k: v for k, v in remaining.items() if k not in list_fields and k not in numeric_dict_fields and k not in scalar_fields}

    if scalar_fields:
        cols = st.columns(min(len(scalar_fields), 4))
        for i, (key, value) in enumerate(scalar_fields.items()):
            with cols[i % len(cols)]:
                st.metric(key, value)

    for key, values in numeric_dict_fields.items():
        st.markdown(f"**{key}**")
        series = pd.Series(values, name=key).sort_values(ascending=False)
        st.dataframe(series, use_container_width=True)
        st.bar_chart(series)

    for key, rows in list_fields.items():
        st.markdown(f"**{key}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if other_fields:
        with st.expander("Other result fields"):
            st.json(other_fields)


def render_calculations(bundle: EvidenceBundle) -> None:
    rows = [
        {"description": c.description, "operation": c.operation, "inputs": c.inputs, "output": c.output}
        for c in bundle.calculations
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_missing_evidence(bundle: EvidenceBundle) -> None:
    rows = [
        {
            "what": item.what,
            "reason": item.reason,
            "reason_code": item.reason_code.value,
            "citation": f"{item.citation.filename} / {item.citation.section}" if item.citation else "",
        }
        for item in bundle.missing_evidence
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_sources(bundle: EvidenceBundle) -> None:
    for source in bundle.sources:
        with st.expander(f"{source.filename} -- {source.section}"):
            st.markdown(f"> {source.snippet}")


def render_timeline(trace: RunTrace) -> None:
    for step in trace.steps:
        with st.expander(f"{step.step}. [{step.type}] {step.name} -- {step.duration_ms} ms"):
            st.markdown("**arguments**")
            st.json(step.arguments)
            st.markdown("**result_summary**")
            st.json(step.result_summary)


def render_llm_usage(trace: RunTrace) -> None:
    if trace.model_calls:
        rows = [
            {
                "provider": call.provider,
                "model": call.model,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "estimated_cost_usd": call.estimated_cost_usd,
                "latency_ms": call.latency_ms,
            }
            for call in trace.model_calls
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"total estimated cost: {trace.estimated_cost_usd}")
    else:
        st.info("Deterministic fallback used -- no LLM call was made for this question.")


def main() -> None:
    st.set_page_config(page_title="Finance Assistant", layout="wide")
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("bundle", None)
    st.session_state.setdefault("trace", None)

    st.title("Finance Assistant")

    with st.sidebar:
        st.header("Settings")
        model_override = st.text_input("Model override (optional)", value="", help="Same as the CLI's --model flag")
        st.header("Example questions")
        for question_text in EXAMPLE_QUESTIONS:
            st.button(
                question_text,
                key=f"example::{question_text}",
                on_click=_set_question,
                args=(question_text,),
                use_container_width=True,
            )

    settings = load_settings(model=model_override or None)
    if settings.has_credential():
        st.success(f"LLM credential detected -- using `{settings.llm_model}`.")
    else:
        checked = ", ".join(f"`{var}`" for var in SUPPORTED_CREDENTIAL_ENV_VARS)
        other_present = settings.other_credentials_present()
        if other_present:
            found = ", ".join(f"`{var}`" for var in other_present)
            st.warning(
                f"`LLM_MODEL={settings.llm_model}` needs `{settings.credential_env_var()}`, which is not "
                f"set (found {found} instead) -- using the deterministic keyword fallback. Set `LLM_MODEL` "
                "to match the credential you have, or add the missing one."
            )
        else:
            st.warning(
                f"No credential found for any supported provider -- checked {checked} -- "
                "using the deterministic keyword fallback, same as the CLI without a credential."
            )

    with st.form("question_form"):
        st.text_area("Ask a question", key="question", height=100)
        submitted = st.form_submit_button("Ask")

    if submitted and st.session_state["question"].strip():
        with st.spinner("Answering..."):
            bundle, trace = answer_question(
                st.session_state["question"],
                data_dir=config.DATA_DIR,
                documents_dir=config.DATA_DIR / "documents",
                model=model_override or None,
            )
        st.session_state["bundle"] = bundle
        st.session_state["trace"] = trace

    bundle: EvidenceBundle | None = st.session_state["bundle"]
    trace: RunTrace | None = st.session_state["trace"]

    if bundle is not None and trace is not None:
        render_status_badge(bundle.status)

        if bundle.status in (AnswerStatus.REFUSED, AnswerStatus.PARTIAL):
            render_coverage_block(bundle)

        st.subheader("Answer")
        if bundle.status == AnswerStatus.REFUSED:
            st.error(bundle.refusal_reason)
        elif bundle.status == AnswerStatus.NEEDS_CLARIFICATION:
            st.markdown("This question needs clarification before it can be answered:")
            for option in bundle.clarification_options:
                st.markdown(f"- {option}")
        elif bundle.status == AnswerStatus.ERROR:
            st.error("The run ended in an error -- see warnings below for details.")
        elif bundle.result is not None:
            render_result(bundle.result, bundle.intent)

        if bundle.calculations:
            st.subheader("Key evidence")
            render_calculations(bundle)

        if bundle.assumptions:
            st.subheader("Assumptions")
            for assumption in bundle.assumptions:
                st.markdown(f"- {assumption}")

        if bundle.warnings:
            st.subheader("Warnings")
            for warning in bundle.warnings:
                st.warning(warning)

        if bundle.missing_evidence:
            st.subheader("Missing evidence")
            render_missing_evidence(bundle)

        if bundle.sources:
            st.subheader("Sources")
            render_sources(bundle)

        st.subheader("How I got this")
        with st.expander("Timeline"):
            render_timeline(trace)

        st.subheader("LLM usage")
        render_llm_usage(trace)

        with st.expander("Raw trace JSON"):
            st.json(trace.model_dump(mode="json"))

    st.divider()
    st.caption(BOUNDARY_PRINCIPLE)


if __name__ == "__main__":
    main()
