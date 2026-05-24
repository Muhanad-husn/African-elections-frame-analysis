"""Offline tests for the production runner (``scripts/run_production.py``).

The production run is a multi-hour, unattended API pass, so its non-API logic —
the ``accepted`` flag, resumability (skipping already-done rows), per-election
staging, and checkpointing — is worth a regression test. ``classify_batch`` is
monkeypatched with a deterministic fake; no network, no cost.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_production.py"


def _load_run_production():
    spec = importlib.util.spec_from_file_location("run_production", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_result(frames, primary, confidence):
    classification = SimpleNamespace(
        frames=frames,
        primary_frame=primary,
        confidence=confidence,
        abstained=not frames,
        rationale="fake rationale",
    )
    return SimpleNamespace(classification=classification, model_used="fake/deepseek")


def _make_clean(tmp_path: Path, n: int = 4) -> Path:
    """A tiny stand-in for articles_clean.parquet."""
    df = pd.DataFrame(
        {
            "GKGRECORDID": [f"id_{i}" for i in range(n)],
            "election": ["kenya_2022", "kenya_2022", "nigeria_2023", "nigeria_2023"][:n],
            "outlet_origin": ["African", "International", "African", "International"][:n],
            "DATE": pd.to_datetime(["2022-08-09"] * n),
            "SourceCommonName": [f"outlet{i}.com" for i in range(n)],
            "text_snippet": [f"snippet {i}" for i in range(n)],
        }
    )
    path = tmp_path / "articles_clean.parquet"
    df.to_parquet(path, index=False)
    return path


def test_accepted_flag_and_schema(tmp_path, monkeypatch):
    rp = _load_run_production()
    clean = _make_clean(tmp_path, n=4)
    out = tmp_path / "articles_classified.parquet"

    # Deterministic per-id outcomes: framed-high (accept), framed-low (reject),
    # abstain (reject), and a hard failure (error row, accepted False).
    canned = {
        "id_0": _fake_result(["economy"], "economy", 0.80),   # accepted
        "id_1": _fake_result(["process"], "process", 0.60),   # framed but below 0.75
        "id_2": _fake_result([], None, 0.20),                 # abstained
    }

    def fake_batch(items, **kwargs):
        out_results = []
        for aid, _text in items:
            if aid in canned:
                out_results.append(SimpleNamespace(article_id=aid, ok=True, result=canned[aid], error=None))
            else:  # id_3 -> hard failure captured (not raised)
                out_results.append(SimpleNamespace(article_id=aid, ok=False, result=None, error="boom"))
        return out_results

    monkeypatch.setattr(rp, "classify_batch", fake_batch)

    rp.run_production(clean_path=clean, out_path=out, threshold=0.75, chunk_size=10)

    df = pd.read_parquet(out).set_index("GKGRECORDID")
    assert len(df) == 4
    assert bool(df.loc["id_0", "accepted"]) is True
    assert bool(df.loc["id_1", "accepted"]) is False  # below threshold
    assert bool(df.loc["id_2", "accepted"]) is False  # abstained
    assert bool(df.loc["id_3", "accepted"]) is False  # error
    assert df.loc["id_2", "pred_abstained"]
    assert df.loc["id_3", "pred_error"] == "boom"
    # Carried analysis columns survive.
    assert set(["election", "outlet_origin", "DATE", "SourceCommonName"]).issubset(df.columns)


def test_resumability_skips_done_rows(tmp_path, monkeypatch):
    rp = _load_run_production()
    clean = _make_clean(tmp_path, n=4)
    out = tmp_path / "articles_classified.parquet"

    seen: list[str] = []

    def fake_batch(items, **kwargs):
        res = []
        for aid, _text in items:
            seen.append(aid)
            res.append(
                SimpleNamespace(
                    article_id=aid, ok=True,
                    result=_fake_result(["economy"], "economy", 0.9), error=None,
                )
            )
        return res

    monkeypatch.setattr(rp, "classify_batch", fake_batch)

    # First pass: 2 rows via --elections filter.
    rp.run_production(clean_path=clean, out_path=out, elections=["kenya_2022"], chunk_size=10)
    assert sorted(seen) == ["id_0", "id_1"]
    assert len(pd.read_parquet(out)) == 2

    # Second pass over the WHOLE corpus: only the 2 not-yet-done rows are classified.
    seen.clear()
    rp.run_production(clean_path=clean, out_path=out, chunk_size=10)
    assert sorted(seen) == ["id_2", "id_3"]
    assert len(pd.read_parquet(out)) == 4

    # Third pass: full no-op (nothing re-classified).
    seen.clear()
    rp.run_production(clean_path=clean, out_path=out, chunk_size=10)
    assert seen == []


def test_missing_clean_raises(tmp_path, monkeypatch):
    rp = _load_run_production()
    with pytest.raises(FileNotFoundError):
        rp.run_production(clean_path=tmp_path / "nope.parquet", out_path=tmp_path / "o.parquet")
