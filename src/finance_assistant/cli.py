"""python -m finance_assistant.cli <question.json | free text>.

Structured JSON mode -- `{"intent": "...", "question": "...", "params": {...}}`
-- needs no LLM credential and resolves an intent directly via
`orchestration.plans.REGISTRY`. Free-text mode routes the question through
`orchestration.orchestrator.answer_question`, which makes exactly one LLM
call when a credential is available and falls back to a deterministic
keyword interpreter otherwise (see `orchestration.interpreter`).

Mode is chosen by file suffix, not file existence: a path ending in
`.json` is always JSON mode -- even if it doesn't exist, that's still a
"question file not found" usage error, never a reinterpretation as free
text (several tests below rely on exactly that). Everything else is free
text. The orchestration import for free-text mode is lazy, inside its own
branch, so pure JSON usage never touches orchestration code at all --
"JSON mode needs no credential" stays structurally true, not incidental.

`REFUSED`/`PARTIAL`/`NEEDS_CLARIFICATION` are legitimate answers, not CLI
failures -- both modes return 0 for all of them. `ERROR` (only reachable
from free-text mode: a broken LLM call, or an exceeded model-call/cost
ceiling) returns 1, alongside the existing usage/dependency/parameter
error cases.

Known limitation: `duplicate_payment_check`'s `rules` parameter
(`DuplicateDetectionRules`) is a typed Python object -- neither JSON nor
an LLM-extractable field can supply it, so the workflow's own default
always applies in both modes.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _exit_with_missing_dependency(exc: ModuleNotFoundError) -> None:
    print(
        f"error: '{exc.name}' is not installed in this interpreter ({sys.executable}) "
        f'— run: .venv\\Scripts\\pip install -e ".[dev]"  then: '
        f".venv\\Scripts\\python.exe -m finance_assistant.cli <question.json>",
        file=sys.stderr,
    )
    sys.exit(1)


try:
    from finance_assistant import config
    from finance_assistant.evidence.models import AnswerStatus
    from finance_assistant.evidence.render import render_bundle_text
    from finance_assistant.evidence.trace import build_trace, write_trace
    from finance_assistant.orchestration.intents import Intent
    from finance_assistant.orchestration.plans import REGISTRY, IntentSpec, PlanResolutionError, build_workflow_kwargs
except ModuleNotFoundError as exc:
    _exit_with_missing_dependency(exc)


class CliError(Exception):
    """A user-facing CLI usage error -- caught in main(), never a raw traceback."""


def _load_question(path: Path) -> dict:
    if not path.is_file():
        raise CliError(f"question file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"question file {path} is not valid JSON: {exc}") from exc
    for key in ("intent", "question", "params"):
        if key not in data:
            raise CliError(f"question file {path} is missing required key {key!r}")
    return data


def _resolve_intent(raw: str) -> tuple[Intent, IntentSpec]:
    valid = ", ".join(sorted(i.value for i in REGISTRY))
    try:
        intent = Intent(raw)
    except ValueError:
        raise CliError(f"unknown intent {raw!r} — valid intents: {valid}") from None
    spec = REGISTRY.get(intent)
    if spec is None:
        raise CliError(f"intent {raw!r} has no workflow registered — valid intents: {valid}")
    return intent, spec


def _run_json_mode(question_path: str, data_dir: Path, traces_dir: Path) -> int:
    try:
        question = _load_question(Path(question_path))
        _, spec = _resolve_intent(question["intent"])
        kwargs = build_workflow_kwargs(spec, question.get("params") or {}, data_dir)
    except (CliError, PlanResolutionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    date_basis = kwargs.get("date_field", config.DEFAULT_FINANCIAL_DATE_FIELD)

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    bundle = spec.workflow(**kwargs)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    trace = build_trace(
        question=question["question"],
        bundle=bundle,
        started_at=started_at,
        duration_ms=duration_ms,
        date_basis=date_basis,
    )
    trace_path = write_trace(trace, traces_dir)

    print(render_bundle_text(bundle))
    print(f"\ntrace written to: {trace_path}", file=sys.stderr)
    return 0


def _run_free_text_mode(question: str, data_dir: Path, traces_dir: Path, model: str | None) -> int:
    try:
        from finance_assistant.orchestration.orchestrator import answer_question
    except ModuleNotFoundError as exc:
        _exit_with_missing_dependency(exc)
        raise  # unreachable -- _exit_with_missing_dependency always calls sys.exit

    bundle, trace = answer_question(question, data_dir=data_dir, documents_dir=data_dir / "documents", model=model)
    trace_path = write_trace(trace, traces_dir)

    print(render_bundle_text(bundle))
    print(f"\ntrace written to: {trace_path}", file=sys.stderr)
    return 1 if bundle.status == AnswerStatus.ERROR else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finance_assistant.cli",
        description="Answer one of the eight analytical questions from a structured JSON file "
        "or free natural-language text, and write a RunTrace JSON.",
    )
    parser.add_argument(
        "question",
        type=str,
        help='a *.json question file ({"intent": "...", "question": "...", "params": {...}}), '
        "or free natural-language text",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="defaults to finance_assistant.config.DATA_DIR")
    parser.add_argument("--traces-dir", type=Path, default=None, help="defaults to finance_assistant.config.TRACES_DIR")
    parser.add_argument("--model", default=None, help="override LLM_MODEL for this run (free-text mode only)")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or config.DATA_DIR
    traces_dir = args.traces_dir or config.TRACES_DIR

    if Path(args.question).suffix.lower() == ".json":
        return _run_json_mode(args.question, data_dir, traces_dir)
    return _run_free_text_mode(args.question, data_dir, traces_dir, args.model)


if __name__ == "__main__":
    sys.exit(main())
