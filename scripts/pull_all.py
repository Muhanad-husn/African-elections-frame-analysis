"""Pull the full ±30-day GDELT GKG windows for all four elections.

Designed to be run as a one-off before the cleaning stage — the pull is
I/O-bound (multi-GB, hours of wall-clock time on a residential connection) and
not something you want to live inside an interactive session. Idempotent:
re-running picks up where it left off.

Usage:
    python scripts/pull_all.py                          # all four elections
    python scripts/pull_all.py kenya_2022               # one election
    python scripts/pull_all.py --days 7                 # narrower window (debug)
    python scripts/pull_all.py kenya_2022 nigeria_2023  # a subset

Expected disk usage (compressed zips): ~30 GB per election at days=30.
"""

from __future__ import annotations

import argparse
import sys

from elections_frames.data import ELECTIONS, pull_gkg_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "elections",
        nargs="*",
        help=f"Elections to pull (default: all). Valid keys: {', '.join(ELECTIONS)}",
    )
    parser.add_argument("--days", type=int, default=30, help="±days around vote day (default: 30)")
    parser.add_argument("--no-progress", action="store_true", help="Suppress tqdm bars")
    args = parser.parse_args(argv)

    keys = args.elections or list(ELECTIONS)
    unknown = [k for k in keys if k not in ELECTIONS]
    if unknown:
        parser.error(f"Unknown election keys: {unknown}. Valid: {list(ELECTIONS)}")

    overall: dict[str, int] = {"downloaded": 0, "cached": 0, "missing": 0}
    for key in keys:
        print(f"\n=== {ELECTIONS[key].name} ({key}) — ±{args.days} days ===", flush=True)
        counts = pull_gkg_window(key, days=args.days, progress=not args.no_progress)
        print(
            f"  downloaded={counts['downloaded']}  cached={counts['cached']}  missing={counts['missing']}",
            flush=True,
        )
        for k, v in counts.items():
            overall[k] += v

    print(f"\nTotal across {len(keys)} election(s): {overall}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
