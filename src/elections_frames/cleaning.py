"""Article cleaning: outlet attribution, English filter, relevance filter, dedup.

The four functions land here as a pipeline:

    raw cached GKG zips
        → attribute_outlet_origin (African / International / Edge / Unknown)
        → filter_english (langdetect on URL title-slug + AllNames)
        → filter_relevant (election-specific theme-tag + keyword rule)
        → deduplicate (URL canonicalization + MinHash on metadata-derived snippet)
        → articles_clean.parquet

A streaming orchestrator (``run_pipeline_election``) is provided because the raw
cache is ~30 GB per election; loading all zips at once is not feasible.

Three of the four functions are the output of a five-part decision block in
``notebooks/02_main.ipynb`` — read the notebook for the rationale
behind specific rules, thresholds, and edge-case verdicts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm.auto import tqdm

from elections_frames.data import (
    ELECTIONS,
    EXTERNAL_DIR,
    KEEP_COLUMNS,
    RAW_DIR,
    Election,
    _read_gkg_zip,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Outlet origin attribution
# ===========================================================================

# African ccTLDs (and the few well-known SLDs that aren't ccTLD-marked).
# Reference: ISO 3166-1 alpha-2 for African states.
# `.africa` is the pan-African gTLD (since 2017).
AFRICAN_CCTLDS: frozenset[str] = frozenset({
    "ng", "co.ng", "com.ng",          # Nigeria
    "ke", "co.ke",                     # Kenya
    "sn",                              # Senegal
    "za", "co.za", "org.za",           # South Africa
    "ma", "tn", "eg", "com.eg",        # North Africa
    "gh", "com.gh",                    # Ghana
    "ug", "co.ug", "tz", "co.tz",      # East Africa
    "rw", "zm", "zw", "co.zw",         # Southern East/Central
    "bw", "na", "mw", "mu", "mg",
    "dz", "ly", "sd",                  # North African / Sahel
    "cm", "ci", "bj", "tg", "bf",      # West/Central francophone
    "ml", "ne", "td", "cf", "ga",
    "cg", "cd",                        # Congos
    "et", "so", "er", "dj",            # Horn
    "lr", "sl", "gm", "cv", "gw", "st",
    "gq", "ao", "mz", "sz", "ls", "ss",
    "km", "sc",
    "africa",                          # gTLD
})

# Curated list of well-known African outlets on `.com`/`.org` (no African ccTLD).
# These are matched on bare-domain substring. Path-sensitive overrides for the
# four named "Edge" outlets are handled separately.
KNOWN_AFRICAN_OUTLETS_NON_CCTLD: frozenset[str] = frozenset({
    "premiumtimesng.com",
    "punchng.com",
    "vanguardngr.com",
    "thisdaylive.com",
    "dailytrust.com",
    "guardian.ng",
    "saharareporters.com",
    "thecable.ng",
    "nation.africa",
    "the-star.co.ke",
    "capitalfm.co.ke",
    "tuko.co.ke",
    "businessdailyafrica.com",
    "mg.co.za",
    "news24.com",
    "iol.co.za",
    "timeslive.co.za",
    "dailymaverick.co.za",
    "businesslive.co.za",
    "ewn.co.za",
    "allafrica.com",
    "africanews.com",
    "semafor.com/africa",   # path-sensitive; semafor.com proper = International
    "theafricareport.com",
    "africa-confidential.com",
    "africa.com",
    "seneweb.com",
    "lesoleil.sn",
    "dakaractu.com",
    "leral.net",
})

# Edge-case URL patterns. **Important:** on inspecting cached GDELT URLs, three
# of the four originally-hypothesized edge desks (AJE, Reuters, RFI English) do
# not separate region in their URL path — only BBC reliably encodes Africa as a
# URL token (``bbc.com/news/world-africa-NNN`` with hyphens, not slashes). And
# RFI's French Africa service is detectable via ``rfi.fr/fr/afrique``. We code
# what URL signal actually supports; for AJE / Reuters / English-RFI we fall
# back to publisher-origin classification (= International).
#
# All ``Edge_*`` labels roll up to International in the headline analysis
# (publisher-origin rule). The desk-staffing alternative is the sensitivity
# check in ``03_robustness.ipynb``.
EDGE_OUTLETS: tuple[tuple[str, str], ...] = (
    ("bbc.com/news/world-africa", "Edge_BBC_Africa"),
    ("bbc.com/news/africa", "Edge_BBC_Africa"),
    ("rfi.fr/fr/afrique", "Edge_RFI_Afrique"),
    ("rfi.fr/en/africa", "Edge_RFI_Afrique"),
)


def collapse_edge_to_international(origin: pd.Series) -> pd.Series:
    """Fold ``Edge_*`` labels into ``International`` for the headline analysis.

    The headline analysis uses publisher-origin attribution; the robustness
    notebook re-runs the analysis without this collapse to test sensitivity.
    """
    return origin.where(~origin.str.startswith("Edge"), "International")


def _tld_chain(host: str) -> list[str]:
    """Yield candidate TLD suffixes from longest to shortest.

    >>> _tld_chain("example.co.ng")
    ['co.ng', 'ng']
    """
    parts = host.lower().split(".")
    if len(parts) < 2:
        return []
    return [".".join(parts[i:]) for i in range(1, len(parts))]


def _classify_by_host(host: str) -> str:
    """Return African / International for a bare hostname (no path)."""
    host = host.lower().lstrip(".").removeprefix("www.")
    if not host:
        return "Unknown"
    if any(host.endswith("." + tld) or host == tld for tld in AFRICAN_CCTLDS):
        return "African"
    if any(host == known or host.endswith("." + known) for known in KNOWN_AFRICAN_OUTLETS_NON_CCTLD):
        return "African"
    return "International"


def attribute_outlet_origin(
    df: pd.DataFrame,
    source_col: str = "SourceCommonName",
    url_col: str = "DocumentIdentifier",
) -> pd.DataFrame:
    """Attribute each row's outlet origin.

    Rule (refined version):

    1. **Path-sensitive edge-case match** against the URL (``DocumentIdentifier``).
       The four named edge outlets (BBC Africa, AJE Africa, Reuters Africa, RFI
       Afrique) are labeled ``Edge_<short>``. Editorial-control verdict:
       Edge → International for headline analysis; sensitivity in robustness.
    2. **TLD match** on the source host (African ccTLDs + ``.africa``) → African.
    3. **Known-outlet match** for African outlets on ``.com``/``.net``/``.org`` → African.
    4. Otherwise → International.

    Rows with no usable host go to ``Unknown``.
    """
    out = df.copy()
    src = out[source_col].fillna("").astype(str).str.lower()
    url = out[url_col].fillna("").astype(str).str.lower()

    origin = pd.Series("International", index=out.index, dtype="object")

    # Step 1: edge-case path match (highest precedence — overrides everything).
    edge_hit = pd.Series(False, index=out.index)
    for path, label in EDGE_OUTLETS:
        mask = url.str.contains(path, regex=False, na=False) & ~edge_hit
        origin.loc[mask] = label
        edge_hit |= mask

    # Step 2 + 3: classify the rest by host.
    remaining = ~edge_hit
    if remaining.any():
        # SourceCommonName is GDELT's "best display name" for the source — typically
        # the bare domain. Strip protocol/path if present.
        hosts = src[remaining].apply(lambda s: urlparse("//" + s).hostname or s)
        classified = hosts.apply(_classify_by_host)
        # If host is empty, fall back to URL netloc.
        empty = hosts == ""
        if empty.any():
            url_hosts = url[remaining].apply(lambda u: urlparse(u).hostname or "")
            fallback = url_hosts.apply(_classify_by_host)
            classified = classified.where(~empty, fallback)
        origin.loc[remaining] = classified

    # Unknown for rows with no usable signal at all.
    no_signal = remaining & (src == "") & (url == "")
    origin.loc[no_signal] = "Unknown"

    out["outlet_origin"] = origin
    return out


# ===========================================================================
# English-language filter
# ===========================================================================

# URL slugs commonly contain the article title in kebab-case — a free
# language signal. AllNames also gives us proper nouns (English-friendly).
_SLUG_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def _url_title_slug(url: str) -> str:
    """Extract the title-slug portion of a URL as space-separated words."""
    if not url:
        return ""
    try:
        path = urlparse(url).path
    except ValueError:
        return ""
    # Take the last non-empty path segment (typically the slug).
    segments = [s for s in path.split("/") if s and "." not in s]
    if not segments:
        return ""
    slug = segments[-1]
    tokens = _SLUG_TOKEN_RE.findall(slug)
    return " ".join(tokens).lower()


def _build_lang_probe(row: pd.Series) -> str:
    """Build text used for langdetect: URL slug + leading names + leading themes."""
    parts = [_url_title_slug(str(row.get("DocumentIdentifier", "") or ""))]
    names = str(row.get("V21AllNames", "") or "")
    if names:
        # V21AllNames format: "Name1,offset;Name2,offset;..." — take first ~5 names.
        first_names = [chunk.split(",")[0] for chunk in names.split(";")[:5]]
        parts.append(" ".join(first_names))
    themes = str(row.get("V2EnhancedThemes", "") or "")
    if themes:
        # V2EnhancedThemes: "THEME1,offset;THEME2,offset;..." — themes are coded
        # in English (e.g. ELECTION, GOV_ELECTION_RIGGING) so they help a probe
        # that's otherwise empty, but we lowercase them and replace underscores
        # so langdetect treats them as English text.
        first_themes = [chunk.split(",")[0] for chunk in themes.split(";")[:5]]
        parts.append(" ".join(t.lower().replace("_", " ") for t in first_themes))
    return " ".join(p for p in parts if p).strip()


def filter_english(
    df: pd.DataFrame,
    min_probe_chars: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop non-English rows. Returns ``(filtered_df, counts)``.

    Strategy: build a per-row probe string from URL title-slug +
    V21AllNames + V2EnhancedThemes, run langdetect, keep ``en``. Rows whose
    probe is shorter than ``min_probe_chars`` are **kept** (langdetect on
    very short strings is unreliable; we err toward inclusion since the
    relevance filter downstream will be more decisive).

    ``counts`` keys: ``n_in``, ``n_kept``, ``n_dropped_non_english``, ``n_probe_too_short``.
    """
    from langdetect import DetectorFactory, LangDetectException
    from langdetect import detect as _detect

    DetectorFactory.seed = seed  # deterministic langdetect

    probes = df.apply(_build_lang_probe, axis=1)
    too_short = probes.str.len() < min_probe_chars

    def _safe_detect(text: str) -> str:
        try:
            return _detect(text)
        except LangDetectException:
            return "unknown"

    # Only run langdetect on rows with enough probe text.
    langs = pd.Series("unknown", index=df.index, dtype="object")
    long_enough = probes.loc[~too_short]
    if len(long_enough) > 0:
        langs.loc[long_enough.index] = long_enough.apply(_safe_detect)

    keep = too_short | (langs == "en")
    filtered = df.loc[keep].reset_index(drop=True)
    counts = {
        "n_in": int(len(df)),
        "n_kept": int(keep.sum()),
        "n_dropped_non_english": int((~keep).sum()),
        "n_probe_too_short": int(too_short.sum()),
    }
    return filtered, counts


# ===========================================================================
# Relevance filter
# ===========================================================================

# GDELT theme codes related to elections. The election-specific keyword set
# lives on the ``Election`` dataclass in ``data.py``.
ELECTION_THEME_PATTERNS: tuple[str, ...] = (
    "ELECTION",                   # generic
    "GOV_ELECTION",
    "POLITICAL_PARTY",            # often co-occurs with election framing
    "DEMOCRACY",
    "GOV_INTERNAL_POLITICS",
)


_TEXT_COLS_FOR_RELEVANCE: tuple[str, ...] = (
    "DocumentIdentifier", "V21AllNames", "V2EnhancedThemes", "SourceCommonName",
)


def filter_relevant(
    df: pd.DataFrame,
    election: Election | str,
    method: str = "hybrid",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep rows about the named election. Vectorized + word-boundary matching.

    Three methods:

    - ``"theme"`` — ``V2EnhancedThemes`` (or ``V1Themes``) contains an
      election-related theme tag (``ELECTION``, ``GOV_ELECTION``,
      ``POLITICAL_PARTY``, ``DEMOCRACY``, ``GOV_INTERNAL_POLITICS``). Generic
      "election as a topic" signal; election-blind.
    - ``"keyword"`` — URL/names/themes/source contains one of the election's
      keywords as a **whole word** (`\\b`-bounded; substring matching produced
      40-70% false-positive recall during the probe). Election-specific.
    - ``"hybrid"`` — theme AND keyword. Highest precision; default.

    See the relevance-filter decision block in ``02_main.ipynb``.
    """
    if isinstance(election, str):
        election = ELECTIONS[election]

    if method not in ("theme", "keyword", "hybrid"):
        raise ValueError(f"Unknown method {method!r}")

    # Word-boundary keyword pattern, case-insensitive.
    kw_pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in election.keywords) + r")\b"
    text_blob = (
        df[list(_TEXT_COLS_FOR_RELEVANCE)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    has_keyword = text_blob.str.contains(kw_pattern, regex=True, case=False, na=False)

    # Theme: substring in themes columns (theme codes are uppercase tokens).
    themes_blob = (
        df[["V2EnhancedThemes", "V1Themes"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    theme_pattern = "|".join(re.escape(p) for p in ELECTION_THEME_PATTERNS)
    has_theme = themes_blob.str.contains(theme_pattern, regex=True, na=False)

    if method == "theme":
        keep = has_theme
    elif method == "keyword":
        keep = has_keyword
    else:  # hybrid
        keep = has_theme & has_keyword

    filtered = df.loc[keep].reset_index(drop=True)
    counts = {
        "n_in": int(len(df)),
        "n_kept": int(keep.sum()),
        "n_theme_only": int((has_theme & ~has_keyword).sum()),
        "n_keyword_only": int((~has_theme & has_keyword).sum()),
        "n_both": int((has_theme & has_keyword).sum()),
        "n_neither": int((~has_theme & ~has_keyword).sum()),
    }
    return filtered, counts


# ===========================================================================
# Deduplication
# ===========================================================================

_TRACKING_QUERY_KEYS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "ref_url",
    "amp", "_amp", "from",
})


def canonicalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase host, strip tracking params, drop fragment."""
    if not url or not isinstance(url, str):
        return ""
    try:
        p = urlparse(url.strip())
    except ValueError:
        return ""
    host = (p.hostname or "").lower().removeprefix("www.")
    path = p.path.rstrip("/")
    # Drop tracking query params; keep order of the rest.
    if p.query:
        kept = [
            kv for kv in p.query.split("&")
            if kv and kv.split("=", 1)[0].lower() not in _TRACKING_QUERY_KEYS
        ]
        query = "&".join(kept)
    else:
        query = ""
    canonical = f"{host}{path}"
    if query:
        canonical += f"?{query}"
    return canonical


def build_text_snippet(row: pd.Series) -> str:
    """Assemble GKG-metadata-only text snippet for classifier input + dedup.

    Per the project's no-scrape decision (Decision Log #1): we do not scrape live HTML.
    The classifier sees URL title-slug + top names + top themes only.
    """
    parts = []
    slug = _url_title_slug(str(row.get("DocumentIdentifier", "") or ""))
    if slug:
        parts.append(f"Title: {slug}")
    names = str(row.get("V21AllNames", "") or "")
    if names:
        first_names = [chunk.split(",")[0] for chunk in names.split(";")[:15]]
        first_names = [n for n in first_names if n]
        if first_names:
            parts.append("Names: " + ", ".join(first_names))
    themes = str(row.get("V2EnhancedThemes", "") or "")
    if themes:
        first_themes = [chunk.split(",")[0] for chunk in themes.split(";")[:15]]
        first_themes = [t for t in first_themes if t]
        if first_themes:
            parts.append("Themes: " + ", ".join(first_themes))
    return " | ".join(parts)


def _shingles(text: str, k: int = 5) -> set[str]:
    """k-character shingles for MinHash."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def deduplicate(
    df: pd.DataFrame,
    threshold: float = 0.8,
    text_col: str = "text_snippet",
    url_col: str = "DocumentIdentifier",
    num_perm: int = 128,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Two-pass dedup: URL canonicalization, then MinHashLSH on text snippet.

    Keeps the **first** row in each cluster (stable wrt input order — pre-sort
    the DataFrame by ``DATE`` upstream if you want the earliest publication).
    """
    from datasketch import MinHash, MinHashLSH

    n_in = len(df)

    # ----- Pass 1: URL canonicalization -----
    canon = df[url_col].fillna("").astype(str).apply(canonicalize_url)
    # Empty canon never matches itself — keep all empty-canon rows.
    is_dup_url = canon.duplicated(keep="first") & (canon != "")
    after_url = df.loc[~is_dup_url].reset_index(drop=True)
    n_after_url = len(after_url)

    # ----- Pass 2: MinHash on text snippet -----
    if text_col not in after_url.columns:
        snippets = after_url.apply(build_text_snippet, axis=1)
    else:
        snippets = after_url[text_col].fillna("").astype(str)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}
    drop_idx: set[int] = set()

    for i, text in enumerate(snippets):
        if not text:
            continue
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(text):
            m.update(sh.encode("utf-8"))
        # Query *before* inserting — first occurrence wins, later near-duplicates drop.
        candidates = lsh.query(m)
        if candidates:
            drop_idx.add(i)
            continue
        lsh.insert(f"row-{i}", m)
        minhashes[i] = m

    keep_mask = ~after_url.index.isin(drop_idx)
    deduped = after_url.loc[keep_mask].reset_index(drop=True)

    counts = {
        "n_in": int(n_in),
        "n_after_url_dedup": int(n_after_url),
        "n_url_duplicates": int(n_in - n_after_url),
        "n_after_text_dedup": int(len(deduped)),
        "n_text_duplicates": int(n_after_url - len(deduped)),
    }
    return deduped, counts


# ===========================================================================
# Streaming probe + pipeline orchestrator
# ===========================================================================

@dataclass
class PipelineCounts:
    """Per-stage row counts for one election's pipeline run.

    Stage order in ``run_pipeline_election`` is raw → relevance → english → dedup
    (relevance first so langdetect runs on the small surviving set, not the firehose).
    """
    election_key: str
    n_raw: int
    n_after_relevant: int
    n_after_english: int
    n_after_dedup: int


def iter_cached_zips(
    election_key: str,
    stride: int | None = None,
    limit: int | None = None,
) -> Iterable[Path]:
    """Yield cached GKG zip paths for one election.

    ``stride`` — yield every Nth zip (for probe sampling).
    ``limit`` — cap total yielded.
    """
    election_dir = RAW_DIR / election_key
    paths = sorted(election_dir.glob("*.gkg.csv.zip"))
    if stride and stride > 1:
        paths = paths[::stride]
    if limit:
        paths = paths[:limit]
    return paths


def load_probe(
    election_key: str,
    stride: int = 96,
    limit: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Load a strided probe sample of cached GKG data for one election.

    Default stride=96 → one 15-min slot per day → ~61 slots per ±30-day
    election window. Cheap enough to iterate decision-block diagnostics
    interactively.
    """
    paths = list(iter_cached_zips(election_key, stride=stride, limit=limit))
    iterator: Iterable[Path] = (
        tqdm(paths, desc=f"probe {election_key}", unit="zip") if progress else paths
    )
    frames = [_read_gkg_zip(p) for p in iterator]
    if not frames:
        return pd.DataFrame(columns=[*KEEP_COLUMNS, "election"])
    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    df = df.dropna(subset=["DATE"]).reset_index(drop=True)
    df["election"] = election_key
    return df


def run_pipeline_election(
    election_key: str,
    relevance_method: str = "hybrid",
    dedup_threshold: float = 0.8,
    batch_size: int = 200,
    write_to: Path | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, PipelineCounts]:
    """Stream-process one election's cached zips through the full cleaning pipeline.

    Reads zips in batches of ``batch_size`` (≈ 2 days at one zip / 15 min),
    applies attribute → english → relevance per batch, accumulates the
    relevant rows, then runs dedup once on the accumulated set.

    Returns the cleaned DataFrame and a ``PipelineCounts`` record. If
    ``write_to`` is given, writes parquet to that path.
    """
    election = ELECTIONS[election_key]
    paths = list(iter_cached_zips(election_key))
    if not paths:
        raise RuntimeError(f"No cached zips for {election_key}")

    accumulated: list[pd.DataFrame] = []
    n_raw_total = 0
    n_after_english_total = 0
    n_after_relevant_total = 0

    n_batches = (len(paths) + batch_size - 1) // batch_size
    batch_iter = range(n_batches)
    if progress:
        batch_iter = tqdm(batch_iter, desc=f"pipeline {election_key}", unit="batch")

    for b in batch_iter:
        batch_paths = paths[b * batch_size:(b + 1) * batch_size]
        frames = [_read_gkg_zip(p) for p in batch_paths]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        n_raw_total += len(df)
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
        df = df.dropna(subset=["DATE"])
        if df.empty:
            continue
        df["election"] = election_key
        df = attribute_outlet_origin(df)
        # Relevance filter FIRST — vectorized regex, removes ~99% of firehose
        # rows. langdetect would otherwise dominate runtime on the unfiltered set.
        df, _ = filter_relevant(df, election, method=relevance_method)
        n_after_relevant_total += len(df)
        if df.empty:
            continue
        df, _ = filter_english(df)
        n_after_english_total += len(df)
        if len(df):
            df["text_snippet"] = df.apply(build_text_snippet, axis=1)
            accumulated.append(df)

    if not accumulated:
        empty = pd.DataFrame(columns=[*KEEP_COLUMNS, "election", "outlet_origin", "text_snippet"])
        counts = PipelineCounts(election_key, n_raw_total, n_after_relevant_total,
                                n_after_english_total, 0)
        return empty, counts

    combined = pd.concat(accumulated, ignore_index=True)
    # Sort by DATE so dedup keeps the earliest publication of each cluster.
    combined = combined.sort_values("DATE").reset_index(drop=True)
    deduped, _ = deduplicate(combined, threshold=dedup_threshold)

    counts = PipelineCounts(
        election_key=election_key,
        n_raw=n_raw_total,
        n_after_relevant=n_after_relevant_total,
        n_after_english=n_after_english_total,
        n_after_dedup=len(deduped),
    )

    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        deduped.to_parquet(write_to, index=False)

    return deduped, counts


# ===========================================================================
# Stratified eval-set sampling
# ===========================================================================

def _week_bucket(article_date: pd.Timestamp, vote_date: pd.Timestamp) -> str:
    """Map an article's date to one of 5 week-buckets around the vote date.

    Buckets: ``pre4``, ``pre2``, ``vote_week``, ``post2``, ``post4``.
    """
    days = (article_date.tz_convert("UTC") - vote_date.tz_convert("UTC")).days
    if days < -14:
        return "pre4"
    if days < -7:
        return "pre2"
    if days <= 7:
        return "vote_week"
    if days <= 21:
        return "post2"
    return "post4"


def sample_eval_candidates(
    df: pd.DataFrame,
    target_total: int = 250,
    n_strata: int | None = None,
    fold_edge_into_international: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample of articles for hand-labeling.

    Strata: election × outlet_origin (folded) × week-bucket-around-vote. With
    4 elections × 2 origins × 5 week-buckets = 40 strata; target_total=250
    yields ~6 per stratum. Strata smaller than the per-stratum allocation
    contribute everything they have.

    Returns a DataFrame with the candidate columns expected by
    ``notebooks/eval_labeling.ipynb`` — including blank
    ``frame_labels`` / ``labeler_notes`` / ``labeled`` columns.
    """
    work = df.copy()
    if fold_edge_into_international:
        work["origin_folded"] = collapse_edge_to_international(work["outlet_origin"])
    else:
        work["origin_folded"] = work["outlet_origin"]

    # Week bucket per row — apply per-election to avoid a NaN-key crash if any
    # row's `election` field is unexpectedly missing.
    vote_dates = {k: pd.Timestamp(e.date) for k, e in ELECTIONS.items()}
    work["week_bucket"] = None
    for key, vote_day in vote_dates.items():
        mask = work["election"] == key
        work.loc[mask, "week_bucket"] = work.loc[mask, "DATE"].apply(
            lambda d, vd=vote_day: _week_bucket(d, vd)
        )

    keys = ["election", "origin_folded", "week_bucket"]
    n_strata_actual = work.groupby(keys).ngroups
    per_stratum = max(1, target_total // n_strata_actual)

    sampled_indices: list = []
    for _, group in work.groupby(keys):
        take = min(per_stratum, len(group))
        sampled_indices.extend(group.sample(n=take, random_state=seed).index.tolist())

    sampled = work.loc[sampled_indices].reset_index(drop=True)

    # If under-target (small strata didn't fill), top up from un-sampled rows.
    if len(sampled) < target_total:
        deficit = target_total - len(sampled)
        leftover = work.drop(index=sampled_indices, errors="ignore")
        if len(leftover):
            top_up = leftover.sample(n=min(deficit, len(leftover)), random_state=seed + 1)
            sampled = pd.concat([sampled, top_up], ignore_index=True)

    # Schema for the labeling UI.
    sampled["frame_labels"] = [[] for _ in range(len(sampled))]
    sampled["labeler_notes"] = ""
    sampled["too_thin"] = False
    sampled["labeled"] = False
    sampled["labeled_at"] = pd.NaT
    return sampled.reset_index(drop=True)
