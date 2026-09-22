"""Fase J -- python scripts/benchmark_providers.py

Runs the eight case questions, paraphrased (never the literal
evals/questions.yaml wording -- see tests/test_orchestration_interpreter.py's
_PARAPHRASES for the same "never calibrate against the eval fixture
strings" rule applied to the keyword fallback), against every configured
*target*. A target is a litellm "provider/model" string -- the comparison
axis here is the model, not the provider: this environment only has one
credential (GROQ_API_KEY), and a single Groq key already reaches several
models of very different size, which proves the same invariant a
multi-provider comparison would. Targets for other providers stay in the
list and show up as clean, reasoned skips -- proof the mechanism already
supports them, not just Groq.

Per target x question this reports: the classified intent (and whether it
matches expected), the extracted parameters (and whether they match
expected), input/output tokens, estimated cost ("unknown" when litellm has
no price for that model), latency, and the final bundle status.

The aggregate metric that is the actual point of the exercise: across
targets that agree on what was asked (same intent, same extracted
parameters), `bundle.result` must be byte-identical, because no target
in this system ever does arithmetic -- workflows/*.py is 100% deterministic
pandas/Python and never imports litellm or any LLM client. A target that
misrouted or mis-extracted parameters answered a *different* question, so
its result is excluded from that comparison (already flagged separately by
the intent/params columns) rather than reported as a false "numeric
divergence".

Ad hoc script: not part of pytest, not part of evals/run_evals.py --
matches how scripts/profile_data.py and evals/run_evals.py themselves have
no dedicated test file and aren't auto-invoked by each other. Run
directly: `python scripts/benchmark_providers.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from finance_assistant import config
from finance_assistant.orchestration.intents import Intent
from finance_assistant.orchestration.interpreter import LiteLLMClient, LLMCompletion
from finance_assistant.orchestration.orchestrator import answer_question
from finance_assistant.orchestration.settings import load_settings

DEFAULT_TARGETS_FILE = config.CONFIG_DIR / "benchmark_targets.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmark"

# The full set of IntentRequest fields a question's text can ever state
# explicitly (see orchestration.intents.IntentRequest) -- anything not
# listed in a BenchmarkQuestion's expected_params is expected to stay None.
PARAM_FIELDS: tuple[str, ...] = (
    "quarter",
    "year",
    "year_current",
    "year_prior",
    "date_start",
    "date_end",
    "top_n",
    "perimeter_basis",
)


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    expected_intent: Intent
    expected_params: dict[str, object] = field(default_factory=dict)

    def expected_value(self, field_name: str) -> object:
        return self.expected_params.get(field_name)


# Paraphrased, one per case question id (see evals/questions.yaml for
# the literal originals) -- different wording/word order throughout.
# Literal parameter tokens (a quarter, a count) are kept only where the
# paraphrase genuinely states them, exactly like the real questions do.
QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        id="q1_opex_by_cost_centre",
        question="Can you break down operating expenses by cost centre for Q2?",
        expected_intent=Intent.OPEX_BY_COST_CENTRE,
        expected_params={"quarter": "Q2"},
    ),
    BenchmarkQuestion(
        id="q2_travel_comparison",
        question="Compare our T&E spend this year against last year.",
        expected_intent=Intent.TRAVEL_COMPARISON,
        expected_params={},
    ),
    BenchmarkQuestion(
        id="q3_consolidated_spend",
        question="What's our total consolidated spend for Q3, expressed in US dollars?",
        expected_intent=Intent.CONSOLIDATED_SPEND,
        expected_params={"quarter": "Q3"},
    ),
    BenchmarkQuestion(
        id="q4_top_vendors",
        question="List the 10 vendors we spend the most with.",
        expected_intent=Intent.TOP_VENDORS,
        expected_params={"top_n": 10},
    ),
    BenchmarkQuestion(
        id="q5_budget_variance",
        question="Which cost centres are furthest over budget in Q3, and what's the main driver?",
        expected_intent=Intent.BUDGET_VARIANCE,
        expected_params={"quarter": "Q3"},
    ),
    BenchmarkQuestion(
        id="q6_te_policy_check",
        question="Which transactions violate our T&E expense policy rules?",
        expected_intent=Intent.TE_POLICY_CHECK,
        expected_params={},
    ),
    BenchmarkQuestion(
        id="q7_headcount_cost_per_fte",
        question="What is our personnel cost on a per-FTE basis?",
        expected_intent=Intent.HEADCOUNT_COST_PER_FTE,
        expected_params={},
    ),
    BenchmarkQuestion(
        id="q8_duplicate_payment_check",
        question="Did we accidentally double-pay any vendor for the same expense?",
        expected_intent=Intent.DUPLICATE_PAYMENT_CHECK,
        expected_params={},
    ),
]


class _CapturingLLMClient:
    """Wraps the real LiteLLMClient and stashes the LLMCompletion from the
    last call, so the benchmark reads intent/params/tokens/cost/latency
    straight from the same LLM call `answer_question` used to build the
    bundle -- never a second, possibly-inconsistent call."""

    def __init__(self) -> None:
        self._client = LiteLLMClient()
        self.last_completion: LLMCompletion | None = None

    def complete(self, *, model, system, user, response_model):
        self.last_completion = self._client.complete(model=model, system=system, user=user, response_model=response_model)
        return self.last_completion


@dataclass
class TargetQuestionResult:
    target: str
    question_id: str
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None
    intent_classified: str | None = None
    intent_match: bool | None = None
    params_extracted: dict[str, object] | None = None
    params_match: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | str | None = None
    latency_ms: int | None = None
    bundle_status: str | None = None
    result: dict | None = None

    @property
    def comparable(self) -> bool:
        return not self.skipped and not self.error and bool(self.intent_match) and bool(self.params_match)


def _load_targets(args: argparse.Namespace) -> list[str]:
    if args.targets:
        return [t.strip() for t in args.targets.split(",") if t.strip()]
    data = yaml.safe_load(args.targets_file.read_text(encoding="utf-8"))
    return list(data["targets"])


def _run_one(target: str, question: BenchmarkQuestion) -> TargetQuestionResult:
    settings = load_settings(model=target)
    if not settings.has_credential():
        return TargetQuestionResult(
            target=target,
            question_id=question.id,
            skipped=True,
            skip_reason=f"no credential for {target} (set {settings.credential_env_var()} to enable)",
        )

    client = _CapturingLLMClient()
    try:
        bundle, _trace = answer_question(question.question, model=target, llm_client=client)
    except Exception as exc:  # a bug in orchestration itself, not an LLM-call failure (those are caught below)
        return TargetQuestionResult(target=target, question_id=question.id, error=f"{type(exc).__name__}: {exc}")

    completion = client.last_completion
    if completion is None:
        # orchestrator.answer_question catches LLM-call exceptions itself and
        # returns an ERROR bundle instead of raising -- surface its real reason.
        reason = "; ".join(bundle.warnings) if bundle.warnings else f"LLM call failed (bundle status={bundle.status.value})"
        return TargetQuestionResult(target=target, question_id=question.id, error=reason)

    extracted = {f: getattr(completion.parsed, f) for f in PARAM_FIELDS}
    expected = {f: question.expected_value(f) for f in PARAM_FIELDS}

    return TargetQuestionResult(
        target=target,
        question_id=question.id,
        intent_classified=completion.parsed.intent.value,
        intent_match=completion.parsed.intent == question.expected_intent,
        params_extracted=extracted,
        params_match=extracted == expected,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        estimated_cost_usd=completion.estimated_cost_usd,
        latency_ms=completion.latency_ms,
        bundle_status=bundle.status.value,
        result=bundle.result,
    )


@dataclass
class DivergenceGroup:
    result: dict | None
    targets: list[str]


@dataclass
class DivergenceReport:
    question_id: str
    comparable_target_count: int
    identical: bool
    groups: list[DivergenceGroup]


def _check_divergence(question_id: str, rows: list[TargetQuestionResult]) -> DivergenceReport:
    comparable = [r for r in rows if r.comparable]
    groups: list[DivergenceGroup] = []
    for r in comparable:
        match = next((g for g in groups if g.result == r.result), None)
        if match is not None:
            match.targets.append(r.target)
        else:
            groups.append(DivergenceGroup(result=r.result, targets=[r.target]))
    return DivergenceReport(
        question_id=question_id,
        comparable_target_count=len(comparable),
        identical=len(groups) <= 1,
        groups=groups,
    )


@dataclass
class Summary:
    targets_configured: int
    targets_executed: int
    targets_skipped: int
    targets_errored: int
    questions_total: int
    questions_compared: int
    divergences_found: int

    def headline(self) -> str:
        bits = [f"{self.targets_executed}/{self.targets_configured} target(s) ejecutado(s)"]
        if self.targets_skipped:
            bits.append(f"{self.targets_skipped} omitido(s) por falta de credencial")
        if self.targets_errored:
            bits.append(f"{self.targets_errored} rechazado(s) por el proveedor (credencial configurada, ver detalle)")
        if self.questions_compared == 0:
            verdict = "sin comparaciones posibles (ningun par de targets coincidio en intent+parametros)"
        elif self.divergences_found == 0:
            verdict = "las cifras no dependen del modelo"
        else:
            verdict = "SE ENCONTRARON DIVERGENCIAS NUMERICAS -- revisar"
        return (
            ", ".join(bits)
            + f", {self.questions_compared}/{self.questions_total} pregunta(s) comparada(s), "
            + f"{self.divergences_found} divergencia(s) numerica(s): {verdict}."
        )


def _build_summary(targets: list[str], results: list[TargetQuestionResult], reports: list[DivergenceReport]) -> Summary:
    executed = {r.target for r in results if not r.skipped and not r.error}
    skipped = {r.target for r in results if r.skipped}
    # A target with a real credential can still fail per-question (e.g. a
    # provider 429 on quota) without ever being "skipped" -- that's a
    # configured-but-rejected outcome, distinct from not-configured-at-all,
    # and stays counted here even if the same target also succeeded on
    # other questions (had_error is not exclusive of executed).
    had_error = {r.target for r in results if r.error}
    questions_compared = sum(1 for rep in reports if rep.comparable_target_count >= 2)
    divergences = sum(1 for rep in reports if rep.comparable_target_count >= 2 and not rep.identical)
    return Summary(
        targets_configured=len(targets),
        targets_executed=len(executed),
        targets_skipped=len(skipped),
        targets_errored=len(had_error),
        questions_total=len(reports),
        questions_compared=questions_compared,
        divergences_found=divergences,
    )


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


DETAIL_HEADERS = ["target", "question", "intent", "intent_ok", "params_ok", "tok_in", "tok_out", "cost_usd", "latency_ms", "status"]


def _truncate(text: str, limit: int = 160) -> str:
    # Console-display-only: a provider error (e.g. a 429 body) can run to
    # several KB, which would blow up the fixed-width table. The JSON
    # report always keeps the untruncated message.
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _detail_rows(results: list[TargetQuestionResult]) -> list[list[str]]:
    rows = []
    for r in results:
        if r.skipped:
            rows.append([r.target, r.question_id, "-", "-", "-", "-", "-", "-", "-", f"SKIP (no credential): {_truncate(r.skip_reason or '')}"])
        elif r.error:
            rows.append([r.target, r.question_id, "-", "-", "-", "-", "-", "-", "-", f"ERROR (provider rejected): {_truncate(r.error)}"])
        else:
            cost = f"{r.estimated_cost_usd:.6f}" if isinstance(r.estimated_cost_usd, float) else str(r.estimated_cost_usd)
            rows.append(
                [
                    r.target,
                    r.question_id,
                    str(r.intent_classified),
                    "yes" if r.intent_match else "NO",
                    "yes" if r.params_match else "NO",
                    str(r.prompt_tokens),
                    str(r.completion_tokens),
                    cost,
                    str(r.latency_ms),
                    str(r.bundle_status),
                ]
            )
    return rows


DIVERGENCE_HEADERS = ["question", "comparable_targets", "verdict", "detail"]


def _divergence_rows(reports: list[DivergenceReport]) -> list[list[str]]:
    rows = []
    for rep in reports:
        if rep.comparable_target_count < 2:
            verdict = "N/A (<2 comparable)"
        elif rep.identical:
            verdict = "IDENTICAL"
        else:
            verdict = "DIVERGENT"
        detail = "" if rep.identical else "; ".join(f"[{', '.join(g.targets)}] -> {g.result}" for g in rep.groups)
        rows.append([rep.question_id, str(rep.comparable_target_count), verdict, detail])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase J -- benchmark configured LLM targets against the eight case questions (paraphrased)"
    )
    parser.add_argument("--targets", help="comma-separated provider/model strings; overrides --targets-file")
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=DEFAULT_TARGETS_FILE,
        help=f"YAML file with a top-level `targets:` list (default: {DEFAULT_TARGETS_FILE})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="where to write the JSON report (default: benchmark/target_benchmark_<UTC timestamp>.json)",
    )
    args = parser.parse_args()

    targets = _load_targets(args)
    if not targets:
        print("no targets configured -- nothing to do", file=sys.stderr)
        return 1

    print(f"[benchmark] {len(targets)} target(s) x {len(QUESTIONS)} question(s)")
    print()

    results: list[TargetQuestionResult] = []
    for target in targets:
        for question in QUESTIONS:
            print(f"[{target}] {question.id} ...", end=" ", flush=True)
            row = _run_one(target, question)
            results.append(row)
            if row.skipped:
                print(f"SKIP ({row.skip_reason})")
            elif row.error:
                print(f"ERROR ({row.error})")
            else:
                print(f"{row.bundle_status} intent_ok={row.intent_match} params_ok={row.params_match}")

    print()
    print("=== Detail ===")
    print(_format_table(DETAIL_HEADERS, _detail_rows(results)))
    print()

    reports = [_check_divergence(q.id, [r for r in results if r.question_id == q.id]) for q in QUESTIONS]

    print("=== Divergence (numeric results across targets that agree on intent + parameters) ===")
    print(_format_table(DIVERGENCE_HEADERS, _divergence_rows(reports)))
    print()

    summary = _build_summary(targets, results, reports)
    print(summary.headline())

    output_path = args.output_json or (
        DEFAULT_OUTPUT_DIR / f"target_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets_configured": targets,
        "results": [asdict(r) for r in results],
        "divergence": [asdict(rep) for rep in reports],
        "summary": asdict(summary),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[benchmark] wrote {output_path}")

    any_divergence = summary.divergences_found > 0
    any_error = any(r.error for r in results)
    return 1 if (any_divergence or any_error) else 0


if __name__ == "__main__":
    sys.exit(main())
