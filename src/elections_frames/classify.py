"""NVIDIA NIM classifier wrapper with structured output + cost logging.

Public surface:

- ``FrameClassification`` — Pydantic schema for the structured output
  (``frames``, ``primary_frame``, ``confidence``, ``rationale``).
- ``classify_article(text, ...)`` — call NVIDIA NIM via the OpenAI-compatible
  client. Retries on transient errors; falls back to a secondary model on hard
  failure. Every attempt is logged to ``data/processed/llm_cost.csv``.

Design choice (Session 4): the wrapper is text-source agnostic — it accepts a
``text: str`` argument with no opinion on whether that's an article body, GKG
metadata, or a hybrid. The ``text_snippet`` definition is settled in Session 3's
structured-dataset decision block. Smoke tests here use raw GKG metadata
strings as stand-ins purely to verify the API round-trip.

API key resolution order:
1. ``NVIDIA_API_KEY`` env var (useful for CI).
2. ``../secrets.toml`` (monorepo parent) under ``[NVIDIA] API_KEY``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import tomllib
import uuid
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
from pydantic import BaseModel, Field, ValidationError

from elections_frames import prompts
from elections_frames.data import PROCESSED_DIR, ROOT

logger = logging.getLogger(__name__)

# --- Endpoint + model IDs ----------------------------------------------------
SECRETS_PATH = ROOT.parent / "secrets.toml"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
PRIMARY_MODEL = "deepseek-ai/deepseek-v4-pro"
FALLBACK_MODEL = "minimaxai/minimax-m2.7"

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
    """Structured classifier output. Validated against parsed model JSON."""

    frames: list[Frame] = Field(min_length=1, max_length=6)
    primary_frame: Frame
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


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
def _load_api_key(secrets_path: Path = SECRETS_PATH) -> str:
    """Resolve the NVIDIA API key from env var or ``../secrets.toml``."""
    env = os.getenv("NVIDIA_API_KEY")
    if env:
        return env
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"NVIDIA API key not found. Set NVIDIA_API_KEY or populate {secrets_path}"
        )
    with secrets_path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data["NVIDIA"]["API_KEY"]
    except KeyError as exc:
        raise KeyError(f"{secrets_path} missing [NVIDIA] API_KEY") from exc


def _build_client(api_key: str | None = None) -> OpenAI:
    """Build an OpenAI client pointed at the NVIDIA NIM endpoint."""
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key or _load_api_key())


def _ensure_cost_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(COST_LOG_COLUMNS)


def _append_cost_row(row: dict[str, object], path: Path = COST_LOG_PATH) -> None:
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
    thinking: bool,
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
        extra_body={"chat_template_kwargs": {"thinking": thinking}},
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
    model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    thinking: bool = False,
    max_tokens: int = 1024,
    max_retries: int = 2,
    client: OpenAI | None = None,
    cost_log_path: Path | None = None,
) -> ClassifyResult:
    """Classify one article into the 6-frame taxonomy via NVIDIA NIM.

    Retries on transient errors (rate limit, timeouts, 5xx); on hard failure
    of the primary model, falls back to ``fallback_model``. Every attempt
    (success, retry, or fallback) writes a row to ``llm_cost.csv``.

    The ``text`` argument is intentionally schema-free — the production
    ``text_snippet`` definition is a Session 3 decision.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    cli = client or _build_client()
    log_path = cost_log_path or COST_LOG_PATH
    messages = prompts.get(prompt_version).render_messages(text)

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
                    cli, attempt_model, messages, thinking, max_tokens
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
