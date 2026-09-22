"""R8 — deterministic keyword/section search over the four markdown documents.

No vector DB, no LLM: sections are parsed by `## ` heading and ranked by a
plain term-frequency score. `filenames` has no default anywhere in this
module — every caller must declare which document(s) it searches, so there
is no code path that silently "searches everything" for every question.

`section` is always the exact heading text, verbatim, never slugified —
this matches the `source_section` convention already used elsewhere
(`cost_centres.AppliedNormalization`, `te_policy.PolicyFinding`,
`config/policy_rules.yaml`), so a citation composes uniformly regardless of
which module produced it.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from finance_assistant import config

DOCUMENTS_DIR = config.DATA_DIR / "documents"

KNOWN_DOCUMENT_FILENAMES = frozenset(
    {
        "board_memo_2024_q2.md",
        "travel_expense_policy.md",
        "contract_kestrel.md",
        "contract_northgate_advisory.md",
    }
)

PREAMBLE_SECTION_TITLE = "(preamble)"
SNIPPET_RADIUS_CHARS = 160

_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TOKEN_PATTERN = re.compile(r"\w+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentSection:
    filename: str
    section: str  # exact "## " heading text, verbatim, or PREAMBLE_SECTION_TITLE
    order: int  # 0-based position within the document, for a stable tie-break
    body: str


@dataclass(frozen=True)
class DocumentMatch:
    filename: str
    section: str
    snippet: str
    evidence_id: str
    score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class DocumentSearchResult:
    query: str
    filenames_searched: list[str]
    matches: list[DocumentMatch]


def parse_markdown_sections(text: str, filename: str) -> list[DocumentSection]:
    headings = list(_HEADING_PATTERN.finditer(text))
    if not headings:
        return [DocumentSection(filename=filename, section=PREAMBLE_SECTION_TITLE, order=0, body=text.strip())]

    sections: list[DocumentSection] = []
    order = 0
    if headings[0].start() > 0:
        preamble = text[: headings[0].start()].strip()
        sections.append(DocumentSection(filename=filename, section=PREAMBLE_SECTION_TITLE, order=order, body=preamble))
        order += 1

    for i, heading in enumerate(headings):
        title = heading.group(1)
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        sections.append(DocumentSection(filename=filename, section=title, order=order, body=body))
        order += 1

    return sections


def _evidence_id(filename: str, section: str, order: int) -> str:
    digest = hashlib.sha1(f"{filename}\x1f{section}\x1f{order}".encode("utf-8")).hexdigest()[:12]
    return f"doc:{Path(filename).stem}:{order}:{digest}"


def tokenize(text: str) -> list[str]:
    """Public so callers can compare their own strings against a
    DocumentMatch.matched_terms set using the exact same term definition
    this module scores with (e.g. checking whether a match's matched_terms
    cover the *entire* term set of some other string, not just overlap it)."""
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if len(t) >= 2]


def _word_boundary_pattern(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b")


def _score_section(section: DocumentSection, terms: list[str]) -> tuple[float, list[str]]:
    body_lower = section.body.lower()
    section_lower = section.section.lower()
    score = 0.0
    matched: list[str] = []
    for term in terms:
        pattern = _word_boundary_pattern(term)
        body_hits = len(pattern.findall(body_lower))
        heading_hit = pattern.search(section_lower) is not None
        if body_hits or heading_hit:
            matched.append(term)
        score += body_hits
        if heading_hit:
            score += 2
    return score, matched


def _extract_snippet(body: str, terms: list[str], radius: int = SNIPPET_RADIUS_CHARS) -> str:
    body_lower = body.lower()
    offsets = [_word_boundary_pattern(term).search(body_lower) for term in terms]
    offsets = [m.start() for m in offsets if m is not None]
    if not offsets:
        truncated = body[: radius * 2].strip()
        suffix = "…" if len(body) > radius * 2 else ""
        return _WHITESPACE_PATTERN.sub(" ", truncated) + suffix

    offset = min(offsets)
    start = max(0, offset - radius)
    end = min(len(body), offset + radius)
    excerpt = body[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return prefix + _WHITESPACE_PATTERN.sub(" ", excerpt) + suffix


def search_documents(
    query: str,
    filenames: list[str],
    documents_dir: str | Path | None = None,
    max_results: int = 10,
) -> DocumentSearchResult:
    if not filenames:
        raise ValueError("filenames must be a non-empty list — search_documents never searches all documents by default")
    unknown = [f for f in filenames if f not in KNOWN_DOCUMENT_FILENAMES]
    if unknown:
        raise ValueError(f"unknown document filename(s): {', '.join(unknown)}")

    terms = tokenize(query)
    if not terms:
        raise ValueError(f"query {query!r} has no searchable terms")

    directory = Path(documents_dir) if documents_dir is not None else DOCUMENTS_DIR

    scored: list[tuple[float, DocumentSection, list[str]]] = []
    for filename in filenames:
        text = (directory / filename).read_text(encoding="utf-8")
        for section in parse_markdown_sections(text, filename):
            score, matched = _score_section(section, terms)
            if score > 0:
                scored.append((score, section, matched))

    scored.sort(key=lambda item: (-item[0], item[1].filename, item[1].order))

    matches = [
        DocumentMatch(
            filename=section.filename,
            section=section.section,
            snippet=_extract_snippet(section.body, matched),
            evidence_id=_evidence_id(section.filename, section.section, section.order),
            score=score,
            matched_terms=matched,
        )
        for score, section, matched in scored[:max_results]
    ]

    return DocumentSearchResult(query=query, filenames_searched=list(filenames), matches=matches)
