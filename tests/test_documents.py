"""Tests for finance_assistant.tools.documents (R8).

Concrete values live only in synthetic tmp_path fixtures built via the
write_markdown fixture, never against the real data/documents/*.md files.
Expected scores/sections are always hand-computed, never derived by calling
the function under test.
"""

import pytest

from finance_assistant.tools.documents import (
    KNOWN_DOCUMENT_FILENAMES,
    PREAMBLE_SECTION_TITLE,
    parse_markdown_sections,
    search_documents,
)

_SIMPLE_DOC = """Some Title
Version 1.0 · effective 1 January 2023

## Alpha Section

This section talks about widget pricing and widget delivery.

## Beta Section

This section talks about gadget pricing.
"""

_NO_HEADINGS_DOC = "Just plain prose with no markdown headings anywhere in it."

_DUPLICATE_HEADING_DOC = """## Notes

First notes section about apples.

## Notes

Second notes section about oranges.
"""

_EMPTY_SECTION_DOC = """## First

## Second

Some body text here.
"""


# ---------------------------------------------------------------------------
# parse_markdown_sections
# ---------------------------------------------------------------------------


def test_parse_sections_captures_preamble():
    sections = parse_markdown_sections(_SIMPLE_DOC, "doc.md")

    assert sections[0].section == PREAMBLE_SECTION_TITLE
    assert sections[0].order == 0
    assert "Version 1.0" in sections[0].body


def test_parse_sections_heading_text_is_verbatim_not_slugified():
    text = "## 2. Americas reorganisation\n\nSome body.\n"

    sections = parse_markdown_sections(text, "doc.md")

    assert sections[0].section == "2. Americas reorganisation"


def test_parse_sections_no_headings_returns_single_section():
    sections = parse_markdown_sections(_NO_HEADINGS_DOC, "doc.md")

    assert len(sections) == 1
    assert sections[0].section == PREAMBLE_SECTION_TITLE
    assert sections[0].body == _NO_HEADINGS_DOC.strip()


def test_parse_sections_duplicate_heading_text_kept_as_separate_sections():
    sections = parse_markdown_sections(_DUPLICATE_HEADING_DOC, "doc.md")

    notes = [s for s in sections if s.section == "Notes"]
    assert len(notes) == 2
    assert notes[0].order != notes[1].order
    assert "apples" in notes[0].body
    assert "oranges" in notes[1].body


def test_parse_sections_empty_section_body_is_kept_not_dropped():
    sections = parse_markdown_sections(_EMPTY_SECTION_DOC, "doc.md")

    titles = [s.section for s in sections]
    assert "First" in titles
    first = next(s for s in sections if s.section == "First")
    assert first.body == ""


# ---------------------------------------------------------------------------
# search_documents — validation
# ---------------------------------------------------------------------------


def test_search_documents_requires_filenames_param():
    with pytest.raises(TypeError):
        search_documents("widget")


def test_search_documents_empty_filenames_raises(write_markdown):
    documents_dir = write_markdown("doc_a.md", _SIMPLE_DOC)

    with pytest.raises(ValueError):
        search_documents("widget", filenames=[], documents_dir=documents_dir)


def test_search_documents_unknown_filename_raises(write_markdown):
    documents_dir = write_markdown("doc_a.md", _SIMPLE_DOC)

    with pytest.raises(ValueError):
        search_documents("widget", filenames=["not_a_real_doc.md"], documents_dir=documents_dir)


def test_search_documents_empty_query_raises(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", _SIMPLE_DOC)

    with pytest.raises(ValueError):
        search_documents("!!!", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)


# ---------------------------------------------------------------------------
# search_documents — scoring and ranking
# ---------------------------------------------------------------------------


def test_search_documents_ranks_heading_hit_above_body_only_hit(write_markdown):
    text = "## Widget Zone\n\nSome unrelated body.\n\n## Other\n\nThis mentions widget once.\n"
    documents_dir = write_markdown("board_memo_2024_q2.md", text)

    result = search_documents("widget", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    assert result.matches[0].section == "Widget Zone"


def test_search_documents_scores_reflect_term_frequency(write_markdown):
    text = "## Section A\n\nwidget widget widget\n\n## Section B\n\nwidget\n"
    documents_dir = write_markdown("board_memo_2024_q2.md", text)

    result = search_documents("widget", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    by_section = {m.section: m.score for m in result.matches}
    assert by_section["Section A"] == pytest.approx(3.0)
    assert by_section["Section B"] == pytest.approx(1.0)


def test_search_documents_no_match_returns_empty_list_not_error(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", _SIMPLE_DOC)

    result = search_documents("zzzznonexistentterm", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    assert result.matches == []


def test_search_documents_only_searches_declared_filenames(write_markdown):
    write_markdown("board_memo_2024_q2.md", "## Only Here\n\nunique_term_xyz appears here.\n")
    documents_dir = write_markdown("travel_expense_policy.md", "## Unrelated\n\nnothing special.\n")

    result = search_documents("unique_term_xyz", filenames=["travel_expense_policy.md"], documents_dir=documents_dir)

    assert result.matches == []


# ---------------------------------------------------------------------------
# evidence_id
# ---------------------------------------------------------------------------


def test_evidence_id_deterministic_across_calls(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", _SIMPLE_DOC)

    first = search_documents("widget", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)
    second = search_documents("widget", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    assert first.matches[0].evidence_id == second.matches[0].evidence_id


def test_evidence_id_distinguishes_duplicate_headings(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", _DUPLICATE_HEADING_DOC)

    result = search_documents("apples oranges", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    ids = {m.evidence_id for m in result.matches}
    assert len(ids) == len(result.matches)


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------


def test_snippet_contains_matched_term_when_present_in_body(write_markdown):
    documents_dir = write_markdown("board_memo_2024_q2.md", _SIMPLE_DOC)

    result = search_documents("gadget", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    assert "gadget" in result.matches[0].snippet.lower()


def test_snippet_falls_back_to_truncation_when_body_has_no_hit(write_markdown):
    text = "## Alpha Zeta\n\nThis body never mentions the search term at all.\n"
    documents_dir = write_markdown("board_memo_2024_q2.md", text)

    result = search_documents("zeta", filenames=["board_memo_2024_q2.md"], documents_dir=documents_dir)

    # "zeta" only hits the heading bonus (Alpha Zeta), never the body itself,
    # so the excerpt-around-match logic has nothing to anchor on and falls
    # back to plain truncation of the body.
    assert "zeta" not in text.split("\n\n", 1)[1].lower()
    assert "never mentions" in result.matches[0].snippet


# ---------------------------------------------------------------------------
# Structural smoke test against real data/documents/ — no concrete assertions
# on score/snippet text, only structure.
# ---------------------------------------------------------------------------


def test_structural_smoke_against_real_documents():
    result = search_documents("lodging", filenames=["travel_expense_policy.md"])

    assert len(result.matches) >= 1
    for match in result.matches:
        assert match.filename == "travel_expense_policy.md"
        assert match.section
        assert match.evidence_id
    assert set(result.filenames_searched) == {"travel_expense_policy.md"}


def test_known_document_filenames_matches_real_documents_directory():
    from finance_assistant.tools.documents import DOCUMENTS_DIR

    on_disk = {p.name for p in DOCUMENTS_DIR.glob("*.md")}
    assert on_disk == set(KNOWN_DOCUMENT_FILENAMES)
