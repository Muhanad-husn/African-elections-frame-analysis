"""Run a prompt version against the hand-labeled eval set and save predictions.

Usage:
    python scripts/run_eval.py v1                 # full eval set (250 rows)
    python scripts/run_eval.py v2 --limit 5       # quick connectivity / sanity check
    python scripts/run_eval.py v2 --thinking      # A/B reasoning mode

Reads ``data/external/eval_set.parquet`` (hand-labeled by Muhanad — treated as
immutable input), classifies each row's ``text_snippet`` via the active provider
(OpenRouter), and writes ``data/processed/eval_results_<version>.parquet`` with
the model's frames / primary / confidence / rationale alongside the gold labels.
Every call is logged to ``data/processed/llm_cost.csv`` by ``classify_article``.

This script never writes to ``eval_set.parquet``.
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from elections_frames.classify import classify_batch, get_provider
from elections_frames.data import EXTERNAL_DIR, PROCESSED_DIR

EVAL_SET_PATH = EXTERNAL_DIR / "eval_set.parquet"


def _model_slug(model: str) -> str:
    """Filename-safe tag from an OpenRouter model id, e.g. minimax/minimax-m2.7 -> minimax-m2-7."""
    return re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")


def run_eval(
    version: str,
    *,
    limit: int | None = None,
    thinking: bool = False,
    model: str | None = None,
    max_workers: int = 8,
    save: bool = True,
) -> pd.DataFrame:
    """Classify the eval set with ``version`` (and optionally a non-default ``model``).

    ``model`` lets us A/B a more capable model (e.g. ``minimax/minimax-m2.7``)
    against the active provider's default (``deepseek/deepseek-v4-flash``) on the
    same prompt — a model comparison distinct from prompt iteration. Outputs are
    tagged with the model slug so they never clobber the default-model runs.
    """
    gold = pd.read_parquet(EVAL_SET_PATH)
    if limit:
        gold = gold.head(limit).copy()

    # Tag everything when a non-default model is used (default = active primary).
    default_model = get_provider().primary_model
    is_alt_model = bool(model) and model != default_model
    model_tag = f"_{_model_slug(model)}" if is_alt_model else ""

    items = list(
        zip(gold["GKGRECORDID"].astype(str), gold["text_snippet"].astype(str), strict=True)
    )
    run_id = f"eval_{version}" + ("_think" if thinking else "") + model_tag
    batch = classify_batch(
        items,
        prompt_version=version,
        run_id=run_id,
        model=model,
        thinking=thinking,
        max_workers=max_workers,
    )

    by_id = {b.article_id: b for b in batch}
    records = []
    for gid in gold["GKGRECORDID"].astype(str):
        b = by_id.get(gid)
        if b is None or not b.ok:
            records.append(
                {
                    "pred_frames": None,
                    "pred_primary": None,
                    "pred_confidence": None,
                    "pred_rationale": None,
                    "pred_abstained": None,
                    "model_used": None,
                    "pred_error": (b.error if b else "missing"),
                }
            )
            continue
        c = b.result.classification
        records.append(
            {
                "pred_frames": list(c.frames),
                "pred_primary": c.primary_frame,
                "pred_confidence": c.confidence,
                "pred_rationale": c.rationale,
                "pred_abstained": c.abstained,
                "model_used": b.result.model_used,
                "pred_error": None,
            }
        )

    preds = pd.DataFrame.from_records(records, index=gold.index)
    preds["model_requested"] = model or default_model
    out = pd.concat(
        [
            gold[
                [
                    "GKGRECORDID", "election", "outlet_origin", "origin_folded",
                    "text_snippet", "frame_labels", "too_thin", "labeler_notes",
                ]
            ].reset_index(drop=True),
            preds.reset_index(drop=True),
        ],
        axis=1,
    )
    out["prompt_version"] = version

    if save:
        suffix = f"_{version}" + ("_think" if thinking else "") + model_tag
        path = PROCESSED_DIR / f"eval_results{suffix}.parquet"
        out.to_parquet(path, index=False)
        n_ok = out["pred_error"].isna().sum()
        print(f"Saved {len(out)} rows ({n_ok} ok) -> {path}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("version", help="prompt version, e.g. v1")
    p.add_argument("--limit", type=int, default=None, help="only classify the first N rows")
    p.add_argument("--thinking", action="store_true", help="enable provider reasoning mode")
    p.add_argument(
        "--model",
        default=None,
        help="override the OpenRouter model, e.g. minimax/minimax-m2.7 (default: deepseek flash)",
    )
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eval(
        args.version,
        limit=args.limit,
        thinking=args.thinking,
        model=args.model,
        max_workers=args.max_workers,
    )
