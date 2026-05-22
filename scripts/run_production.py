"""Production classification run — prompt v3 / deepseek-v4-flash over the full corpus.

Classifies every row of ``data/processed/articles_clean.parquet`` (79,372 articles)
into the 6-frame taxonomy via ``classify_batch`` and writes
``data/processed/articles_classified.parquet``.

Production config is **locked** by Session 5 (see ``IMPLEMENTATION_PLAN.md`` Decision
Log #5): prompt ``v3``, model ``deepseek/deepseek-v4-flash``, abstention enabled.

Threshold (Session 6, decision block in ``04_pipeline_eval.ipynb`` §6): the 0.75
confidence elbow is recorded as a **flag, not a filter** — every row is written with
its raw label, ``pred_confidence``, and an ``accepted`` boolean
(``accepted = framed AND confidence >= threshold``). Nothing is dropped, so the
headline analysis (Session 7) and the threshold ±0.05 sensitivity (Session 8) can
both work off this one file.

Cost / time: ~$23 and ~9-18 h at 8-16 workers (≈ 1,652 in / 480 out tokens per row
at Flash rates). Real spend — this script is meant to be launched by the user.

It is **idempotent / resumable**: rows whose ``GKGRECORDID`` already appears in the
output parquet are skipped, and the parquet is checkpointed after every chunk, so a
crash or Ctrl-C only loses the in-flight chunk. Stage by election with ``--elections``:

    python scripts/run_production.py                      # full corpus, all 4 elections
    python scripts/run_production.py --elections kenya_2022   # pilot one election first
    python scripts/run_production.py --limit 20          # tiny smoke (real API calls)
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pandas as pd

from elections_frames.classify import classify_batch, get_provider
from elections_frames.data import ELECTIONS, PROCESSED_DIR

CLEAN_PATH = PROCESSED_DIR / "articles_clean.parquet"
OUT_PATH = PROCESSED_DIR / "articles_classified.parquet"

# Locked production config (Session 5) + threshold (Session 6).
PROMPT_VERSION = "v3"
THRESHOLD = 0.75

# Columns carried into the output so the analysis notebook can compute the headline
# frame distribution without re-joining to articles_clean. (text_snippet is *not*
# duplicated — join back on GKGRECORDID if the raw input is needed.)
CARRY_COLUMNS = ["GKGRECORDID", "election", "outlet_origin", "DATE", "SourceCommonName"]


def _done_ids(out_path: Path) -> set[str]:
    """GKGRECORDIDs already classified (for resumability)."""
    if not out_path.exists():
        return set()
    return set(pd.read_parquet(out_path, columns=["GKGRECORDID"])["GKGRECORDID"].astype(str))


def _records_for_chunk(chunk: pd.DataFrame, batch, threshold: float) -> list[dict]:
    """Map ``classify_batch`` results back onto the chunk rows, in chunk order."""
    by_id = {b.article_id: b for b in batch}
    out: list[dict] = []
    for _, row in chunk.iterrows():
        gid = str(row["GKGRECORDID"])
        carried = {c: row[c] for c in CARRY_COLUMNS}
        b = by_id.get(gid)
        if b is None or not b.ok:
            out.append({
                **carried,
                "pred_frames": None, "pred_primary": None, "pred_confidence": None,
                "pred_rationale": None, "pred_abstained": None, "accepted": False,
                "model_used": None, "pred_error": (b.error if b else "missing"),
            })
            continue
        c = b.result.classification
        accepted = (not c.abstained) and (c.confidence is not None) and (c.confidence >= threshold)
        out.append({
            **carried,
            "pred_frames": list(c.frames),
            "pred_primary": c.primary_frame,
            "pred_confidence": c.confidence,
            "pred_rationale": c.rationale,
            "pred_abstained": c.abstained,
            "accepted": bool(accepted),
            "model_used": b.result.model_used,
            "pred_error": None,
        })
    return out


def _checkpoint(records: list[dict], out_path: Path) -> int:
    """Append ``records`` to the output parquet (read-concat-write). Returns total rows."""
    new = pd.DataFrame.from_records(records)
    if out_path.exists():
        new = pd.concat([pd.read_parquet(out_path), new], ignore_index=True)
    new.to_parquet(out_path, index=False)
    return len(new)


def run_production(
    *,
    version: str = PROMPT_VERSION,
    elections: list[str] | None = None,
    limit: int | None = None,
    threshold: float = THRESHOLD,
    max_workers: int = 8,
    chunk_size: int = 1000,
    clean_path: Path = CLEAN_PATH,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found — run `python scripts/run_cleaning.py` first."
        )
    clean = pd.read_parquet(clean_path)
    if elections:
        clean = clean[clean["election"].isin(elections)].copy()
    if limit:
        clean = clean.head(limit).copy()

    done = _done_ids(out_path)
    todo = clean[~clean["GKGRECORDID"].astype(str).isin(done)].reset_index(drop=True)

    model = get_provider().primary_model
    run_id = "prod_" + uuid.uuid4().hex[:12]
    print(
        f"Production run {run_id} | prompt {version} | model {model} | threshold {threshold}\n"
        f"  corpus={len(clean):,}  already done={len(done):,}  to classify={len(todo):,}"
        f"  ({max_workers} workers, chunk {chunk_size})"
    )
    if todo.empty:
        print("Nothing to do — output is already complete for this selection.")
        return pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start : start + chunk_size]
        items = list(
            zip(chunk["GKGRECORDID"].astype(str), chunk["text_snippet"].astype(str), strict=True)
        )
        batch = classify_batch(
            items, prompt_version=version, run_id=run_id, model=model, max_workers=max_workers
        )
        total = _checkpoint(_records_for_chunk(chunk, batch, threshold), out_path)
        n_err = sum(not b.ok for b in batch)
        print(
            f"  chunk {start // chunk_size + 1}: +{len(chunk)} rows "
            f"({n_err} errors) -> {out_path.name} now {total:,} rows"
        )

    final = pd.read_parquet(out_path)
    sel = final[final["election"].isin(elections)] if elections else final
    n_acc = int(sel["accepted"].sum())
    n_abst = int(sel["pred_abstained"].fillna(False).sum())
    n_err = int(sel["pred_error"].notna().sum())
    print(
        f"\nDone. {len(sel):,} rows for this selection: "
        f"{n_acc:,} accepted (framed & conf>={threshold}), "
        f"{n_abst:,} abstained, {n_err:,} errors."
    )
    return final


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elections", nargs="*", default=None,
                   help=f"Election keys to run; default = all. Choices: {list(ELECTIONS)}")
    p.add_argument("--version", default=PROMPT_VERSION, help="Prompt version (default: v3).")
    p.add_argument("--limit", type=int, default=None, help="Only classify the first N rows.")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"Confidence floor for the `accepted` flag (default: {THRESHOLD}).")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=1000, help="Rows per checkpoint.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_production(
        version=args.version,
        elections=args.elections,
        limit=args.limit,
        threshold=args.threshold,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
    )
