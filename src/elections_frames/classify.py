"""OpenRouter / NVIDIA NIM classifier wrapper with structured output + cost logging.

Public surface:

- ``FrameClassification`` — Pydantic schema for the structured output
  (``frames``, ``primary_frame``, ``confidence``, ``rationale``).
- ``classify_article(text, ...)`` — call the active provider via the
  OpenAI-compatible client. Retries on transient errors; falls back to a
  secondary model on hard failure. Every attempt is logged to
  ``data/processed/llm_cost.csv``.
- ``classify_batch(items, ...)`` — thread-pooled fan-out of ``classify_article``
  over many ``(article_id, text)`` items; per-item failures are captured rather
  than fatal, and results are returned in input order.

Provider: the **active** provider is OpenRouter
(``deepseek/deepseek-v4-flash`` primary, ``minimax/minimax-m2.7`` fallback).
NVIDIA NIM is retained as a switchable provider (``provider="nvidia"``) but is
inactive by default: its free tier ran 23–855 s/call with a ~37% transient-error
rate, which made eval-scale passes (~20 h single-threaded) and the
production run impractical. OpenRouter Flash is the throughput/cost fix.

Design choice: the wrapper is text-source agnostic — it accepts a
``text: str`` argument with no opinion on whether that's an article body, GKG
metadata, or a hybrid. The ``text_snippet`` definition is settled in the
structured-dataset decision block. Smoke tests here use raw GKG metadata
strings as stand-ins purely to verify the API round-trip.

API key resolution order (for the active provider):
1. The provider's env var (``OPENROUTER_API_KEY`` / ``NVIDIA_API_KEY``) — useful for CI.
2. ``../secrets.toml`` (monorepo parent) under the provider's ``[section]`` key.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import tomllib
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError, model_validator

from elections_frames import prompts
from elections_frames.data import PROCESSED_DIR, ROOT

logger = logging.getLogger(__name__)

# --- Providers ---------------------------------------------------------------
SECRETS_PATH = ROOT.parent / "secrets.toml"


@dataclass(frozen=True)
class Provider:
    """Endpoint + model + secrets binding for one OpenAI-compatible backend."""

    name: str
    base_url: str
    env_var: str
    secrets_section: str
    secrets_key: str
    primary_model: str
    fallback_model: str

    def thinking_extra_body(self, thinking: bool) -> dict[str, object]:
        """Provider-specific request body for toggling reasoning/thinking mode.

        NVIDIA NIM uses ``chat_template_kwargs``; OpenRouter uses the unified
        ``reasoning`` parameter and would 400 on ``chat_template_kwargs``. For
        OpenRouter we only send the key when thinking is requested, so the
        common (thinking-off) path stays a plain JSON-mode call.
        """
        if self.name == "nvidia":
            return {"chat_template_kwargs": {"thinking": thinking}}
        return {"reasoning": {"enabled": True}} if thinking else {}


OPENROUTER = Provider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    env_var="OPENROUTER_API_KEY",
    secrets_section="OPENROUTER",
    secrets_key="OPENROUTER_API_KEY",
    primary_model="deepseek/deepseek-v4-flash",
    fallback_model="minimax/minimax-m2.7",
)

# Inactive by default — see module docstring. Kept switchable for an easy revert.
NVIDIA = Provider(
    name="nvidia",
    base_url="https://integrate.api.nvidia.com/v1",
    env_var="NVIDIA_API_KEY",
    secrets_section="NVIDIA",
    secrets_key="API_KEY",
    primary_model="deepseek-ai/deepseek-v4-pro",
    fallback_model="minimaxai/minimax-m2.7",
)

PROVIDERS: dict[str, Provider] = {p.name: p for p in (OPENROUTER, NVIDIA)}

# Active provider for the production run (NVIDIA -> OpenRouter migration).
ACTIVE_PROVIDER = "openrouter"


def get_provider(name: str | None = None) -> Provider:
    """Return the provider config for ``name`` (defaults to ``ACTIVE_PROVIDER``)."""
    key = name or ACTIVE_PROVIDER
    try:
        return PROVIDERS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown provider {key!r}; known: {sorted(PROVIDERS)}") from exc


# Back-compat module constants — resolve to the active provider.
PRIMARY_MODEL = get_provider().primary_model
FALLBACK_MODEL = get_provider().fallback_model

# --- Cost log ----------------------------------------------------------------
COST_LOG_PATH = PROCESSED_DIR / "llm_cost.csv"
COST_LOG_COLUMNS = (
    "timestamp",
    "run_id",
    "prompt_version",
    "model",
    "thinking",
    "article_id",
    "input_tokens",
    "output_tokens",
    "duration_seconds",
    "status",
    "fallback_used",
    "validation_error",
)

# --- Schema ------------------------------------------------------------------
Frame = Literal["security", "economy", "democracy", "identity", "process", "corruption"]
FRAME_VALUES: tuple[str, ...] = (
    "security",
    "economy",
    "democracy",
    "identity",
    "process",
    "corruption",
)


class FrameClassification(BaseModel):
    """Structured classifier output. Validated against parsed model JSON.

    Abstention: ``frames`` may be **empty** to mean "no frame is
    clearly foregrounded" — this matches how the eval set was hand-labeled
    (135/250 rows carry an empty label because the GKG-metadata-only snippet
    is too thin to foreground a frame). When ``frames`` is empty,
    ``primary_frame`` is ``None``. The constraint was loosened from the
    ``min_length=1`` so the classifier can express that abstention;
    prompt ``v1`` never abstains (its instruction defaults thin inputs to
    ``["process"]``), so v1's behavior is unchanged. Out-of-vocabulary frame
    values are still a hard validation error.
    """

    frames: list[Frame] = Field(default_factory=list, max_length=6)
    primary_frame: Frame | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reconcile_primary(self) -> FrameClassification:
        """Keep ``primary_frame`` consistent with ``frames`` (lenient, non-raising).

        Normalizes rather than rejects so a well-formed-but-slightly-inconsistent
        model response still scores rather than burning a retry+fallback:
        - empty ``frames`` -> abstain, force ``primary_frame=None``;
        - non-empty ``frames`` with no primary -> take the first as primary;
        - a primary not listed in ``frames`` -> fold it into ``frames`` (front).
        """
        if not self.frames:
            self.primary_frame = None
        elif self.primary_frame is None:
            self.primary_frame = self.frames[0]
        elif self.primary_frame not in self.frames:
            self.frames = [self.primary_frame, *self.frames]
        return self

    @property
    def abstained(self) -> bool:
        """True when the model foregrounded no frame (empty ``frames``)."""
        return not self.frames


@dataclass
class ClassifyResult:
    """The parsed classification plus per-call metadata."""

    classification: FrameClassification
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    model_used: str
    fallback_used: bool
    raw_content: str


# --- Internals ---------------------------------------------------------------
def _load_api_key(provider: Provider, secrets_path: Path = SECRETS_PATH) -> str:
    """Resolve ``provider``'s API key from its env var or ``../secrets.toml``."""
    env = os.getenv(provider.env_var)
    if env:
        return env
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"{provider.name} API key not found. "
            f"Set {provider.env_var} or populate {secrets_path}"
        )
    with secrets_path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data[provider.secrets_section][provider.secrets_key]
    except KeyError as exc:
        raise KeyError(
            f"{secrets_path} missing [{provider.secrets_section}] {provider.secrets_key}"
        ) from exc


def _build_client(provider: Provider, api_key: str | None = None) -> OpenAI:
    """Build an OpenAI client pointed at ``provider``'s endpoint."""
    return OpenAI(base_url=provider.base_url, api_key=api_key or _load_api_key(provider))


def _ensure_cost_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(COST_LOG_COLUMNS)


# Serializes concurrent cost-log appends from ``classify_batch`` worker threads.
_COST_LOG_LOCK = threading.Lock()


def _append_cost_row(row: dict[str, object], path: Path = COST_LOG_PATH) -> None:
    with _COST_LOG_LOCK:
        _ensure_cost_log(path)
        with path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=COST_LOG_COLUMNS).writerow(
                {k: row.get(k, "") for k in COST_LOG_COLUMNS}
            )


_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


def _attempt_classify(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    extra_body: dict[str, object],
    max_tokens: int,
) -> tuple[str, int, int, float]:
    """One API call. Returns ``(content, input_tokens, output_tokens, duration_s)``."""
    start = time.monotonic()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        top_p=0.95,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    duration = time.monotonic() - start
    content = completion.choices[0].message.content or ""
    usage = completion.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return content, in_tok, out_tok, duration


# --- Public API --------------------------------------------------------------
def classify_article(
    text: str,
    prompt_version: str = "v1",
    article_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = 1024,
    max_retries: int = 2,
    client: OpenAI | None = None,
    cost_log_path: Path | None = None,
) -> ClassifyResult:
    """Classify one article into the 6-frame taxonomy via the active provider.

    ``provider`` defaults to ``ACTIVE_PROVIDER`` (OpenRouter); ``model`` and
    ``fallback_model`` default to that provider's configured models. Retries on
    transient errors (rate limit, timeouts, 5xx); on hard failure of the primary
    model, falls back to ``fallback_model``. Every attempt (success, retry, or
    fallback) writes a row to ``llm_cost.csv``.

    The ``text`` argument is intentionally schema-free — the production
    ``text_snippet`` definition is a cleaning-stage decision.
    """
    prov = get_provider(provider)
    run_id = run_id or uuid.uuid4().hex[:12]
    model = model or prov.primary_model
    fallback_model = fallback_model or prov.fallback_model
    cli = client or _build_client(prov)
    log_path = cost_log_path or COST_LOG_PATH
    messages = prompts.get(prompt_version).render_messages(text)
    extra_body = prov.thinking_extra_body(thinking)

    fallback_used = False
    last_exc: Exception | None = None

    for attempt_model in (model, fallback_model):
        backoff = 1.0
        for attempt in range(max_retries + 1):
            row: dict[str, object] = {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "run_id": run_id,
                "prompt_version": prompt_version,
                "model": attempt_model,
                "thinking": thinking,
                "article_id": article_id or "",
                "fallback_used": fallback_used,
            }
            try:
                content, in_tok, out_tok, dur = _attempt_classify(
                    cli, attempt_model, messages, extra_body, max_tokens
                )
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                row.update(
                    input_tokens=0,
                    output_tokens=0,
                    duration_seconds=0.0,
                    status=f"transient:{type(exc).__name__}",
                    validation_error="",
                )
                _append_cost_row(row, log_path)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break  # exhausted retries on this model -> fall through to fallback
            except (BadRequestError, APIError) as exc:
                last_exc = exc
                row.update(
                    input_tokens=0,
                    output_tokens=0,
                    duration_seconds=0.0,
                    status=f"hard:{type(exc).__name__}",
                    validation_error="",
                )
                _append_cost_row(row, log_path)
                break  # don't retry hard errors; fall through to fallback

            # Parse + validate structured output.
            try:
                parsed = json.loads(content)
                classification = FrameClassification.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_exc = exc
                row.update(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    duration_seconds=round(dur, 3),
                    status="parse_error",
                    validation_error=str(exc)[:300],
                )
                _append_cost_row(row, log_path)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break

            # Success path.
            row.update(
                input_tokens=in_tok,
                output_tokens=out_tok,
                duration_seconds=round(dur, 3),
                status="ok",
                validation_error="",
            )
            _append_cost_row(row, log_path)
            return ClassifyResult(
                classification=classification,
                input_tokens=in_tok,
                output_tokens=out_tok,
                duration_seconds=dur,
                model_used=attempt_model,
                fallback_used=fallback_used,
                raw_content=content,
            )
        fallback_used = True

    raise RuntimeError(
        f"classify_article failed for both {model} and {fallback_model}: {last_exc!r}"
    ) from last_exc


# --- Batch (thread-pooled) ---------------------------------------------------
@dataclass
class BatchResult:
    """One article's batch outcome — exactly one of ``result`` / ``error`` is set."""

    article_id: str
    result: ClassifyResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


def classify_batch(
    items: Iterable[tuple[str, str]],
    *,
    prompt_version: str = "v1",
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = 1024,
    max_retries: int = 2,
    max_workers: int = 8,
    client: OpenAI | None = None,
    cost_log_path: Path | None = None,
    progress: bool = True,
) -> list[BatchResult]:
    """Classify many ``(article_id, text)`` items concurrently via a thread pool.

    Throughput path for the eval/production passes: ``classify_article`` is
    per-article and the OpenAI client is thread-safe, so the only shared mutable
    resource is the ``llm_cost.csv`` append — serialized by ``_COST_LOG_LOCK``.

    Results are returned **in input order**. A per-item hard failure (both models
    exhausted) is captured in ``BatchResult.error`` rather than aborting the run;
    every attempt is still logged to the cost CSV by ``classify_article``. All
    items share one ``run_id`` (so a run groups cleanly in the cost log) and one
    client (one connection pool). Size ``max_workers`` against the provider's
    rate limits — OpenRouter Flash tolerates modest concurrency comfortably.
    """
    work = list(items)
    if not work:
        return []

    run_id = run_id or uuid.uuid4().hex[:12]
    prov = get_provider(provider)
    cli = client or _build_client(prov)
    log_path = cost_log_path or COST_LOG_PATH
    results: list[BatchResult | None] = [None] * len(work)

    def _one(idx: int, article_id: str, text: str) -> tuple[int, BatchResult]:
        try:
            res = classify_article(
                text,
                prompt_version=prompt_version,
                article_id=article_id,
                run_id=run_id,
                provider=provider,
                model=model,
                fallback_model=fallback_model,
                thinking=thinking,
                max_tokens=max_tokens,
                max_retries=max_retries,
                client=cli,
                cost_log_path=log_path,
            )
            return idx, BatchResult(article_id=article_id, result=res)
        except Exception as exc:  # noqa: BLE001 - batch must survive per-item failures
            logger.warning("classify_batch: %s failed: %r", article_id, exc)
            return idx, BatchResult(article_id=article_id, result=None, error=repr(exc))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, i, aid, txt) for i, (aid, txt) in enumerate(work)]
        iterator = as_completed(futures)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(futures), desc=f"classify[{prov.name}]")
        for fut in iterator:
            idx, br = fut.result()
            results[idx] = br

    return [r for r in results if r is not None]
