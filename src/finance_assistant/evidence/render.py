"""Deterministic EvidenceBundle -> plain text renderer. No LLM.

This module decides nothing: `AnswerStatus` and every field on the bundle
are already final by the time a bundle reaches here (evidence/gate.py owns
the decision). Rendering is pure formatting of what the bundle already
contains.

Two callers, both already anticipated by docs/PROMPT_MAESTRO.md: it is the
"keyword fallback renderer" required for when there is no LLM credential
("Sin credencial de LLM... proveer un intérprete de respaldo"), and it is
what evals/run_evals.py's `forbidden_claims` regex checks run against,
since no answer text may ever be produced except from bundle contents.
"""

from finance_assistant.evidence.models import EvidenceBundle


def _render_value(value: object, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            lines.append(f"{pad}(empty)")
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_render_value(val, indent + 1))
            else:
                lines.append(f"{pad}{key}: {val}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{pad}(none)")
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_render_value(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{value}")
    return lines


def render_bundle_text(bundle: EvidenceBundle) -> str:
    lines: list[str] = [
        f"status: {bundle.status.value}",
        f"intent: {bundle.intent.value}",
    ]

    if bundle.refusal_reason:
        lines.append(f"refusal_reason: {bundle.refusal_reason}")

    if bundle.clarification_options:
        lines.append("clarification_options:")
        lines.extend(_render_value(bundle.clarification_options, 1))

    if bundle.assumptions:
        lines.append("assumptions:")
        lines.extend(_render_value(bundle.assumptions, 1))

    if bundle.warnings:
        lines.append("warnings:")
        lines.extend(_render_value(bundle.warnings, 1))

    if bundle.missing_evidence:
        lines.append("missing_evidence:")
        for item in bundle.missing_evidence:
            lines.append(f"  - what: {item.what}")
            lines.append(f"    reason: {item.reason}")
            lines.append(f"    reason_code: {item.reason_code.value}")
            if item.citation:
                lines.append(f"    citation: {item.citation.filename} / {item.citation.section}")

    if bundle.sources:
        lines.append("sources:")
        for source in bundle.sources:
            lines.append(f"  - {source.filename} / {source.section}: {source.snippet}")

    lines.append(
        "coverage: "
        f"selected_rows={bundle.coverage.selected_rows} "
        f"computable_rows={bundle.coverage.computable_rows} "
        f"computable_amount_pct={bundle.coverage.computable_amount_pct}"
    )

    if bundle.result is not None:
        lines.append("result:")
        lines.extend(_render_value(bundle.result, 1))

    if bundle.calculations:
        lines.append("calculations:")
        for calc in bundle.calculations:
            lines.append(f"  - description: {calc.description}")
            lines.append(f"    operation: {calc.operation}")
            lines.append(f"    inputs: {calc.inputs}")
            lines.append(f"    output: {calc.output}")

    return "\n".join(lines)
