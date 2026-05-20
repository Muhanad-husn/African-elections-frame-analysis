"""GDELT 2.0 GKG ingestion + caching + outlet provenance join.

Filled in during Session 2 (see ``IMPLEMENTATION_PLAN.md``). Planned surface:

- ``pull_gkg_window(election, days=30)`` — fetch 15-minute GKG files for the
  ±days window around the election, cache to ``data/raw/<election>/``,
  append file IDs to ``data/raw/manifest.json``.
- ``load_cached(election)`` — read all cached records for one election into
  a DataFrame.
- ``outlet_provenance_join(df, outlets_csv)`` — left-join on
  ``SourceCommonName`` against ``data/external/outlets.csv``; unmatched
  outlets get ``origin = "Unknown"`` and are resolved during the Session 3
  outlet-attribution decision block.
"""

from __future__ import annotations

from pathlib import Path

# Project paths — referenced from notebooks via ``from elections_frames.data import RAW_DIR``.
# Auto-created on import so smoke tests pass before Session 2 fills in the puller.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / ".cache"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
