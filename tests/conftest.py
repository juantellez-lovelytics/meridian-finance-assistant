from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from finance_assistant.orchestration import settings as _settings_module
from finance_assistant.orchestration.interpreter import LLMCompletion
from finance_assistant.orchestration.intents import IntentRequest


@pytest.fixture(autouse=True)
def _no_real_dotenv_reload(monkeypatch):
    """Freezes `settings._DOTENV_LOADED` as already-True for every test.

    `load_settings()` lazily loads the real `.env` on first call, via a
    module-level "only once" flag -- so whichever test happens to run
    first in the session determines whether `.env` gets (re)read *during*
    that test. Since `dotenv.load_dotenv(override=False)` only skips vars
    already present, a test that does `monkeypatch.delenv("GEMINI_API_KEY")`
    before that first load has it silently restored from the real `.env`
    a moment later -- order-dependent and invisible until `.env` holds a
    real credential. Freezing the flag makes every test see exactly the
    ambient/monkeypatched environment, never the real `.env` file.
    """
    monkeypatch.setattr(_settings_module, "_DOTENV_LOADED", True)


@pytest.fixture
def write_csv(tmp_path):
    """Writes a CSV under tmp_path from a header list + row dicts.

    Ensures tests that need concrete, known values never touch the real
    data/ directory — every synthetic dataset lives entirely in tmp_path.
    """

    def _write(filename: str, header: list[str], rows: list[dict]) -> Path:
        df = pd.DataFrame(rows, columns=header)
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def write_markdown(tmp_path):
    """Writes a markdown file under a documents/ subdir of tmp_path.

    Mirrors write_csv's tmp_path-only isolation — tests for tools.documents
    never touch the real data/documents/*.md content.
    """

    def _write(filename: str, text: str) -> Path:
        documents_dir = tmp_path / "documents"
        documents_dir.mkdir(exist_ok=True)
        path = documents_dir / filename
        path.write_text(text, encoding="utf-8")
        return documents_dir

    return _write


@pytest.fixture
def fake_llm_client():
    """Builds an `interpreter.LLMClient` that returns a fixed `IntentRequest`
    without ever importing or calling litellm -- the dependency-injection
    seam that lets orchestration tests exercise the LLM code path with no
    real network call. `model` is deliberately NOT a fixture parameter: like
    the real `LiteLLMClient`, `complete()` echoes back whichever model the
    caller passed it, it never has its own stored default."""

    def _make(
        response: IntentRequest,
        *,
        provider: str = "fake",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        latency_ms: int = 1,
        estimated_cost_usd: float | str = 0.001,
    ):
        @dataclass
        class _FakeLLMClient:
            def complete(self, *, model, system, user, response_model):
                return LLMCompletion(
                    parsed=response,
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    estimated_cost_usd=estimated_cost_usd,
                )

        return _FakeLLMClient()

    return _make
