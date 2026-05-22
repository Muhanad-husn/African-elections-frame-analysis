"""Classifier smoke tests.

Two layers:

1. **Mocked** — runs offline; injects a fake OpenAI client that returns
   canned JSON. Verifies parsing, schema validation, and cost-log writes.
2. **Live** — gated behind ``OPENROUTER_LIVE=1``; hits the real active-provider
   endpoint (OpenRouter by default) with ~5 records from the Kenya 2022 smoke
   pull and asserts the round-trip yields valid ``FrameClassification`` objects.
"""

from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace

import pytest


# --- Mocked ------------------------------------------------------------------
class _FakeCompletions:
    def __init__(self, content: str, in_tok: int = 123, out_tok: int = 45):
        self._content = content
        self._in = in_tok
        self._out = out_tok

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=self._in, completion_tokens=self._out),
        )


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, content: str, **token_kwargs):
        self.chat = _FakeChat(_FakeCompletions(content, **token_kwargs))


def test_classify_mocked_success(tmp_path):
    from elections_frames.classify import classify_article

    canned = json.dumps(
        {
            "frames": ["economy", "democracy"],
            "primary_frame": "economy",
            "confidence": 0.87,
            "rationale": "Lead foregrounds inflation; democratic-norms angle is secondary.",
        }
    )
    client = _FakeClient(canned, in_tok=900, out_tok=80)
    log = tmp_path / "llm_cost.csv"

    result = classify_article(
        text="dummy article",
        article_id="probe_001",
        client=client,
        cost_log_path=log,
    )
    assert result.classification.primary_frame == "economy"
    assert result.classification.frames == ["economy", "democracy"]
    assert result.input_tokens == 900
    assert result.output_tokens == 80
    assert result.fallback_used is False

    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["article_id"] == "probe_001"
    assert int(rows[0]["input_tokens"]) == 900


def test_classify_mocked_invalid_frame_value(tmp_path):
    """Model returns an out-of-vocab frame — Pydantic catches it; we log parse_error."""
    from elections_frames.classify import classify_article

    bad = json.dumps(
        {
            "frames": ["security", "vibes"],  # 'vibes' not in Frame literal
            "primary_frame": "security",
            "confidence": 0.5,
            "rationale": "x",
        }
    )
    client = _FakeClient(bad)
    log = tmp_path / "llm_cost.csv"

    with pytest.raises(RuntimeError):
        # Will retry, hit fallback (same fake content), and ultimately raise.
        classify_article(
            text="dummy",
            client=client,
            cost_log_path=log,
            max_retries=0,
        )

    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) >= 1
    assert all(r["status"] == "parse_error" for r in rows)
    assert any("vibes" in r["validation_error"] for r in rows)


def test_classify_batch_mocked_preserves_order_and_logs(tmp_path):
    """classify_batch fans out, preserves input order, and logs one row per item."""
    from elections_frames.classify import classify_batch

    canned = json.dumps(
        {
            "frames": ["security"],
            "primary_frame": "security",
            "confidence": 0.9,
            "rationale": "stand-in",
        }
    )
    client = _FakeClient(canned)
    log = tmp_path / "llm_cost.csv"

    items = [(f"art_{i}", f"text {i}") for i in range(12)]
    results = classify_batch(
        items,
        client=client,
        cost_log_path=log,
        max_workers=4,
        progress=False,
    )

    assert [r.article_id for r in results] == [aid for aid, _ in items]  # input order
    assert all(r.ok for r in results)
    assert all(r.result.classification.primary_frame == "security" for r in results)

    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == 12  # one ok row per item — no interleaving/corruption
    assert {r["run_id"] for r in rows} == {rows[0]["run_id"]}  # one shared run_id
    assert all(r["status"] == "ok" for r in rows)


def test_classify_batch_mocked_survives_item_failures(tmp_path):
    """A hard per-item failure is captured in BatchResult.error, not raised."""
    from elections_frames.classify import classify_batch

    bad = json.dumps(
        {"frames": ["nope"], "primary_frame": "nope", "confidence": 0.1, "rationale": "x"}
    )
    client = _FakeClient(bad)
    log = tmp_path / "llm_cost.csv"

    items = [(f"art_{i}", f"text {i}") for i in range(5)]
    results = classify_batch(
        items,
        client=client,
        cost_log_path=log,
        max_workers=3,
        max_retries=0,
        progress=False,
    )

    assert len(results) == 5
    assert all(not r.ok for r in results)
    assert all(r.error is not None for r in results)
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert rows and all(r["status"] == "parse_error" for r in rows)


def test_prompt_v1_messages_have_codebook_and_examples():
    """Sanity: the v1 prompt embeds the codebook and at least 2 few-shot examples."""
    from elections_frames import prompts

    v1 = prompts.get("v1")
    messages = v1.render_messages("an article about something")
    assert len(messages) == 2
    sys = messages[0]["content"]
    usr = messages[1]["content"]
    assert "an article about something" in usr
    # Codebook content
    for frame in ("security", "economy", "democracy", "identity", "process", "corruption"):
        assert frame in sys
    # At least 2 worked examples (EXAMPLE 1, EXAMPLE 2)
    assert sys.count("EXAMPLE ") >= 2


# --- Live --------------------------------------------------------------------
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_LIVE"),
    reason="Set OPENROUTER_LIVE=1 to enable the live active-provider round-trip smoke test.",
)
def test_classify_live_smoke():
    """Live: classify 5 GKG records from the Kenya 2022 smoke pull.

    Hits the active provider (OpenRouter by default). Uses GKG metadata strings
    (themes + entities + URL) as the classifier input — this is a Session-4 smoke
    stand-in, not a commitment to the production text_snippet definition (that's
    Session 3's decision).
    """
    from elections_frames.classify import FrameClassification, classify_article
    from elections_frames.data import load_cached

    df = load_cached("kenya_2022")
    if len(df) == 0:
        pytest.skip("No cached Kenya 2022 records; run the smoke pull first.")

    sample = df.head(5)
    results = []
    for _, row in sample.iterrows():
        descriptor = "\n".join(
            f"{k}: {row[k]}" for k in ("DocumentIdentifier", "V2EnhancedThemes", "V21AllNames") if row[k]
        )[:4000]
        result = classify_article(
            text=descriptor,
            article_id=row["GKGRECORDID"],
            run_id="session4_smoke_kenya_2022",
        )
        assert isinstance(result.classification, FrameClassification)
        assert result.input_tokens > 0
        results.append(result)

    assert len(results) == 5
    # At least one classification should have come back; print summary for eyeball review.
    for r in results:
        print(
            f"  {r.classification.primary_frame:10s}  "
            f"conf={r.classification.confidence:.2f}  "
            f"frames={r.classification.frames}  "
            f"({r.input_tokens}+{r.output_tokens} tok, {r.duration_seconds:.1f}s)"
        )
