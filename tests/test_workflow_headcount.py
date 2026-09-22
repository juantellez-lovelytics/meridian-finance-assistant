"""Tests for finance_assistant.workflows.headcount.headcount_cost_per_fte (Q7).

Concrete values live only in synthetic tmp_path fixtures, never against the
real data/documents/*.md content. Confirms the refusal is structural
(always fires) and that its citation comes from a real search, not a
hardcoded string.
"""

from finance_assistant.evidence.models import AnswerStatus, MissingEvidenceReasonCode
from finance_assistant.workflows.headcount import headcount_cost_per_fte


def test_always_refuses_with_missing_fte_denominator_reason(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHeadcount is tracked by HR, not finance.\n")

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.result is None
    assert bundle.missing_evidence[0].reason_code == MissingEvidenceReasonCode.MISSING_FTE_DENOMINATOR


def test_citation_comes_from_real_search_not_hardcoded(write_markdown):
    documents_dir = write_markdown(
        "board_memo_2024_q2.md", "## Some Unrelated Section\n\nnothing to do with the topic.\n\n## Staffing Levels\n\nheadcount figures live in the HR system.\n"
    )

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.sources[0].section == "Staffing Levels"
    assert bundle.missing_evidence[0].citation.section == "Staffing Levels"


def test_refuses_even_when_document_has_no_relevant_section(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", "## Unrelated\n\nnothing about staffing here.\n")

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.missing_evidence[0].citation is None
    assert any("no section" in w for w in bundle.warnings)


def test_result_is_none_and_bundle_construction_does_not_raise(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHR system.\n")

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)

    assert bundle.result is None
    assert bundle.refusal_reason is not None


def test_tool_calls_are_recorded(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", "## Headcount\n\nHeadcount is tracked by HR, not finance.\n")

    bundle = headcount_cost_per_fte(documents_dir=documents_dir)

    assert bundle.tool_calls
    tool_names = {tc.tool for tc in bundle.tool_calls}
    assert tool_names <= {"search_documents"}
    assert "search_documents" in tool_names
    assert all(tc.duration_ms >= 0 for tc in bundle.tool_calls)


# ---------------------------------------------------------------------------
# Structural smoke test against real data/documents/ — no concrete values.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_documents():
    bundle = headcount_cost_per_fte()

    assert bundle.status == AnswerStatus.REFUSED
    assert bundle.sources[0].filename == "board_memo_2024_q2.md"
