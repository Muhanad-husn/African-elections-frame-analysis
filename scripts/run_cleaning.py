"""Run the cleaning pipeline across all four elections.

Stream-processes ``data/raw/<election>/*.gkg.csv.zip`` through
``cleaning.run_pipeline_election`` and writes
``data/processed/articles_clean_<election>.parquet`` plus a combined
``data/processed/articles_clean.parquet``.

This is the multi-hour I/O step after the GDELT pull. Idempotent — re-running
overwrites the per-election parquets. Per-stage row counts are appended to
``data/processed/pipeline_counts.csv``.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from elections_frames.cleaning import run_pipeline_election
from elections_frames.data import ELECTIONS, PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elections", nargs="*", default=list(ELECTIONS),
                        help="Election keys to process; default = all 4.")
    parser.add_argument("--method", default="hybrid",
                        choices=("theme", "keyword", "hybrid"),
                        help="Relevance method (default: hybrid).")
    parser.add_argument("--dedup-threshold", type=float, default=0.8,
                        help="MinHash Jaccard threshold (default: 0.8).")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="GKG zips per batch (default: 200 ≈ 2 days).")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts_rows: list[dict] = []
    election_frames: list[pd.DataFrame] = []

    for key in args.elections:
        print(f"\n=== {ELECTIONS[key].name} ({key}) — {args.method} relevance, dedup@{args.dedup_threshold} ===")
        t0 = time.monotonic()
        out_path = PROCESSED_DIR / f"articles_clean_{key}.parquet"
        df, counts = run_pipeline_election(
            key,
            relevance_method=args.method,
            dedup_threshold=args.dedup_threshold,
            batch_size=args.batch_size,
            write_to=out_path,
            progress=True,
        )
        elapsed = time.monotonic() - t0
        print(f"  raw={counts.n_raw:,} → relevant={counts.n_after_relevant:,}"
              f" → english={counts.n_after_english:,} → dedup={counts.n_after_dedup:,}"
              f"  ({elapsed/60:.1f} min)")
        counts_rows.append({
            "election": key,
            "method": args.method,
            "dedup_threshold": args.dedup_threshold,
            "n_raw": counts.n_raw,
            "n_after_relevant": counts.n_after_relevant,
            "n_after_english": counts.n_after_english,
            "n_after_dedup": counts.n_after_dedup,
            "elapsed_sec": round(elapsed, 1),
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if len(df):
            election_frames.append(df)

    # Combined parquet.
    if election_frames:
        combined = pd.concat(election_frames, ignore_index=True)
        combined_path = PROCESSED_DIR / "articles_clean.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nCombined parquet: {combined_path}  ({len(combined):,} rows)")

    counts_df = pd.DataFrame(counts_rows)
    counts_path = PROCESSED_DIR / "pipeline_counts.csv"
    if counts_path.exists():
        existing = pd.read_csv(counts_path)
        counts_df = pd.concat([existing, counts_df], ignore_index=True)
    counts_df.to_csv(counts_path, index=False)
    print(f"\nPipeline counts: {counts_path}")
    print(counts_df.tail(len(args.elections)).to_string(index=False))


if __name__ == "__main__":
    main()
