"""GDELT 2.0 GKG ingestion + caching + outlet provenance join.

Two responsibilities:

1. Pull GKG records for ±30-day windows around each election, cache zipped
   archives to ``data/raw/<election>/``, track pulled slot IDs in
   ``data/raw/manifest.json``.
2. Load cached records back into a DataFrame and attach outlet provenance
   from ``data/external/outlets.csv``.

The English-relevance filter and dedup happen in ``cleaning.py`` —
this module pulls everything that lands in the time window.
"""

from __future__ import annotations

import csv
import json
import logging
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Project paths. Auto-created on import so smoke tests pass before any pull runs.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / ".cache"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = RAW_DIR / "manifest.json"

GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
SLOT_INTERVAL_MINUTES = 15

# GDELT 2.0 GKG column schema — 27 fields, tab-separated.
# Reference: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
GKG_COLUMNS: tuple[str, ...] = (
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "V1Counts",
    "V21Counts",
    "V1Themes",
    "V2EnhancedThemes",
    "V1Locations",
    "V2EnhancedLocations",
    "V1Persons",
    "V2EnhancedPersons",
    "V1Organizations",
    "V2EnhancedOrganizations",
    "V2Tone",
    "V2EnhancedDates",
    "V2GCAM",
    "V2SharingImage",
    "V21RelatedImages",
    "V21SocialImageEmbeds",
    "V21SocialVideoEmbeds",
    "V21Quotations",
    "V21AllNames",
    "V21Amounts",
    "V21TranslationInfo",
    "V2ExtrasXML",
)

# Columns we retain on load. Dropping GCAM / Extras / embed columns up front cuts
# memory on a full corpus by >50%; downstream consumers can re-load with a wider
# `keep_columns` if needed.
KEEP_COLUMNS: tuple[str, ...] = (
    "GKGRECORDID",
    "DATE",
    "SourceCommonName",
    "DocumentIdentifier",
    "V1Themes",
    "V2EnhancedThemes",
    "V1Locations",
    "V21AllNames",
    "V2Tone",
)


@dataclass(frozen=True)
class Election:
    """One election: filesystem-safe key, display name, vote day (UTC), keywords."""

    key: str
    name: str
    date: datetime
    keywords: tuple[str, ...]


# Election dates verified 2026-05-21 via web search (Wikipedia + IFES + Al Jazeera).
ELECTIONS: dict[str, Election] = {
    # en.wikipedia.org/wiki/2023_Nigerian_presidential_election
    # Bola Tinubu (APC) defeated Atiku Abubakar (PDP) and Peter Obi (LP).
    # Keyword lists broadened after the initial draft under-recalled
    # in Senegal/Kenya (only 42/41 hybrid hits per ±30-day probe slot-per-day).
    # Broadening rationale lives in the relevance-filter decision block of
    # `notebooks/02_main.ipynb`. Keywords are word-boundary-matched
    # case-insensitively (see `cleaning.filter_relevant`).
    "nigeria_2023": Election(
        key="nigeria_2023",
        name="Nigeria 2023 presidential",
        date=datetime(2023, 2, 25, tzinfo=UTC),
        keywords=(
            "Nigerian", "Nigeria",
            "Tinubu", "Bola Tinubu",
            "Atiku", "Atiku Abubakar",
            "Peter Obi",
            "INEC",
            "APC", "PDP", "Labour Party",
        ),
    ),
    # en.wikipedia.org/wiki/2022_Kenyan_general_election
    # William Ruto defeated Raila Odinga.
    "kenya_2022": Election(
        key="kenya_2022",
        name="Kenya 2022 general",
        date=datetime(2022, 8, 9, tzinfo=UTC),
        keywords=(
            "Kenyan", "Kenya",
            "Ruto", "William Ruto",
            "Raila", "Odinga", "Raila Odinga",
            "Uhuru", "Kenyatta",
            "IEBC", "UDA", "Azimio",
        ),
    ),
    # en.wikipedia.org/wiki/2024_Senegalese_presidential_election
    # Originally scheduled 2024-02-25; postponed by President Sall and held on
    # 2024-03-24. Bassirou Diomaye Faye won in the first round. The window is
    # centered on the actual vote day (the postponement itself is part of the
    # story coverage will frame around).
    "senegal_2024": Election(
        key="senegal_2024",
        name="Senegal 2024 presidential",
        date=datetime(2024, 3, 24, tzinfo=UTC),
        keywords=(
            "Senegalese", "Senegal",
            "Faye", "Bassirou", "Diomaye", "Bassirou Diomaye",
            "Sall", "Macky", "Macky Sall",
            "Sonko", "Ousmane Sonko",
            "Amadou Ba",
            "PASTEF",
        ),
    ),
    # en.wikipedia.org/wiki/2024_South_African_general_election
    # ANC lost its parliamentary majority for the first time since 1994.
    "south_africa_2024": Election(
        key="south_africa_2024",
        name="South Africa 2024 general",
        date=datetime(2024, 5, 29, tzinfo=UTC),
        keywords=(
            "South Africa", "South African",
            "Ramaphosa", "Cyril Ramaphosa",
            "Malema", "Julius Malema",
            "Zuma", "Jacob Zuma",
            "ANC", "African National Congress",
            "DA Party", "Democratic Alliance",
            "MK Party", "uMkhonto we Sizwe",
            "EFF", "Economic Freedom Fighters",
        ),
    ),
}


def _iter_slots(start: datetime, end: datetime) -> Iterator[str]:
    """Yield GDELT slot slugs (YYYYMMDDhhmmss) at 15-min intervals in [start, end)."""
    step = timedelta(minutes=SLOT_INTERVAL_MINUTES)
    minute_floor = (start.minute // SLOT_INTERVAL_MINUTES) * SLOT_INTERVAL_MINUTES
    cur = start.replace(minute=minute_floor, second=0, microsecond=0)
    while cur < end:
        yield cur.strftime("%Y%m%d%H%M%S")
        cur += step


def _slot_url(slug: str) -> str:
    return f"{GDELT_BASE_URL}/{slug}.gkg.csv.zip"


def _slot_path(election_key: str, slug: str) -> Path:
    return RAW_DIR / election_key / f"{slug}.gkg.csv.zip"


def _load_manifest() -> dict[str, list[str]]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def _save_manifest(manifest: dict[str, list[str]]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def download_slot(
    election_key: str,
    slug: str,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> str:
    """Download one GDELT 15-min GKG zip; idempotent.

    Returns ``"downloaded"``, ``"cached"`` (already on disk and non-empty), or
    ``"missing"`` (404 — GDELT skipped that slot, which happens occasionally).
    """
    dest = _slot_path(election_key, slug)
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests
    resp = sess.get(_slot_url(slug), stream=True, timeout=timeout)
    if resp.status_code == 404:
        return "missing"
    resp.raise_for_status()

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)
    return "downloaded"


def pull_gkg_window(
    election_key: str,
    days: int = 30,
    max_slots: int | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """Pull GDELT GKG records for ±``days`` around the election vote day.

    Idempotent — already-downloaded zips are not re-fetched. ``max_slots`` caps
    the number of slots considered (useful for smoke tests; default = all
    ``(days*2 + 1) * 96`` slots). Returns counts dict with keys ``downloaded``,
    ``cached``, ``missing``.
    """
    if election_key not in ELECTIONS:
        raise KeyError(f"Unknown election {election_key!r}. Known: {list(ELECTIONS)}")
    election = ELECTIONS[election_key]
    start = election.date - timedelta(days=days)
    end = election.date + timedelta(days=days + 1)  # exclusive — covers vote_day + days_after
    slots = list(_iter_slots(start, end))
    if max_slots is not None:
        slots = slots[:max_slots]

    counts = {"downloaded": 0, "cached": 0, "missing": 0}
    session = requests.Session()

    manifest = _load_manifest()
    pulled: set[str] = set(manifest.get(election_key, []))

    iterator: Iterable[str] = (
        tqdm(slots, desc=f"GKG {election_key}", unit="slot") if progress else slots
    )
    for slug in iterator:
        try:
            result = download_slot(election_key, slug, session=session)
        except requests.RequestException as exc:
            logger.warning("Slot %s failed: %s", slug, exc)
            continue
        counts[result] += 1
        if result in ("downloaded", "cached"):
            pulled.add(slug)

    manifest[election_key] = sorted(pulled)
    _save_manifest(manifest)
    return counts


def _read_gkg_zip(zip_path: Path, keep_columns: tuple[str, ...] = KEEP_COLUMNS) -> pd.DataFrame:
    """Read one cached GKG zip into a DataFrame restricted to ``keep_columns``."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            if not members:
                return pd.DataFrame(columns=list(keep_columns))
            with zf.open(members[0]) as fh:
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    names=GKG_COLUMNS,
                    dtype=str,
                    on_bad_lines="skip",
                    quoting=csv.QUOTE_NONE,
                    encoding="utf-8",
                    encoding_errors="replace",
                    low_memory=False,
                )
    except zipfile.BadZipFile:
        logger.warning("Bad zip skipped: %s", zip_path)
        return pd.DataFrame(columns=list(keep_columns))

    return df[list(keep_columns)]


def load_cached(
    election_key: str,
    keep_columns: tuple[str, ...] = KEEP_COLUMNS,
) -> pd.DataFrame:
    """Read all cached GKG zips for one election into a single DataFrame.

    Parses ``DATE`` to UTC datetime; rows whose date fails to parse are dropped.
    Tags every row with ``election = election_key``.
    """
    if election_key not in ELECTIONS:
        raise KeyError(f"Unknown election {election_key!r}")
    election_dir = RAW_DIR / election_key
    if not election_dir.exists():
        return pd.DataFrame(columns=[*keep_columns, "election"])

    paths = sorted(election_dir.glob("*.gkg.csv.zip"))
    if not paths:
        return pd.DataFrame(columns=[*keep_columns, "election"])

    frames = [_read_gkg_zip(p, keep_columns=keep_columns) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    df = df.dropna(subset=["DATE"]).reset_index(drop=True)
    df["election"] = election_key
    return df


def outlet_provenance_join(
    df: pd.DataFrame,
    outlets_csv: Path | None = None,
) -> pd.DataFrame:
    """Left-join ``df.SourceCommonName`` against ``outlets.csv`` to tag origin.

    Pragmatic implementation: simple bare-domain substring match.
    Unmatched rows get ``outlet_origin = "Unknown"``. the outlet
    attribution decision block refines this (path-sensitive matching for the
    four named edge cases — BBC Africa, AJE Africa desk, Reuters Africa, RFI
    Afrique — and bulk resolution of residual unknowns).
    """
    if outlets_csv is None:
        outlets_csv = EXTERNAL_DIR / "outlets.csv"
    out = df.copy()
    out["outlet_origin"] = "Unknown"

    if not outlets_csv.exists():
        return out

    outlets = pd.read_csv(outlets_csv)
    src_lower = out["SourceCommonName"].fillna("").str.lower()

    for _, row in outlets.iterrows():
        domain = str(row.get("domain", "")).strip().lower()
        if not domain or domain == "nan":
            continue
        bare = domain.split("/", 1)[0]
        mask = src_lower.str.contains(bare, regex=False)
        unset = out["outlet_origin"] == "Unknown"
        out.loc[mask & unset, "outlet_origin"] = str(row["origin"])

    return out
