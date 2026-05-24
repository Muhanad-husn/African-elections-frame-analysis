"""Smoke tests — package imports, data paths exist, and (optionally) a live GDELT pull.

Live network tests are gated behind ``GDELT_LIVE=1`` so default ``pytest``
runs stay offline.
"""

from __future__ import annotations

import os

import pytest


def test_package_imports():
    """The top-level package imports."""
    import elections_frames  # noqa: F401


def test_module_imports():
    """All planned modules import (most are stubs at this stage)."""
    from elections_frames import (  # noqa: F401
        classify,
        cleaning,
        data,
        diagnostics,
        viz,
    )


def test_data_dirs_exist():
    """Auto-created data directories exist on package import."""
    from elections_frames.data import EXTERNAL_DIR, PROCESSED_DIR, RAW_DIR

    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert EXTERNAL_DIR.exists()


def test_diagnostics_helpers_present():
    """The six diagnostic helpers are importable."""
    from elections_frames.diagnostics import (  # noqa: F401
        before_after,
        compare_alternatives,
        distribution_compare,
        distribution_summary,
        missingness_pattern,
        missingness_summary,
    )


def test_prompts_package_present():
    """Versioned prompts package is importable."""
    from elections_frames import prompts  # noqa: F401


def test_elections_metadata():
    """All four planned elections are declared with verified dates."""
    from elections_frames.data import ELECTIONS

    assert set(ELECTIONS) == {
        "nigeria_2023",
        "kenya_2022",
        "senegal_2024",
        "south_africa_2024",
    }
    assert ELECTIONS["nigeria_2023"].date.year == 2023
    assert ELECTIONS["kenya_2022"].date.year == 2022
    assert ELECTIONS["senegal_2024"].date.month == 3  # postponed from Feb to March
    assert ELECTIONS["south_africa_2024"].date.day == 29


def test_iter_slots_at_15min_boundary():
    """The slot iterator yields exactly four slugs per hour at HH:00/15/30/45."""
    from datetime import UTC, datetime

    from elections_frames.data import _iter_slots

    start = datetime(2022, 8, 9, 0, 0, tzinfo=UTC)
    end = datetime(2022, 8, 9, 1, 0, tzinfo=UTC)
    slugs = list(_iter_slots(start, end))
    assert slugs == ["20220809000000", "20220809001500", "20220809003000", "20220809004500"]


@pytest.mark.skipif(
    not os.getenv("GDELT_LIVE"),
    reason="Set GDELT_LIVE=1 to enable the live GDELT round-trip smoke test.",
)
def test_pull_gkg_window_smoke_kenya_2022():
    """Live: pull a few slots for Kenya 2022 vote day and verify load_cached works."""
    from elections_frames.data import load_cached, pull_gkg_window

    counts = pull_gkg_window("kenya_2022", days=0, max_slots=4, progress=False)
    # At least one slot should have been downloaded or already cached (4 attempted).
    assert counts["downloaded"] + counts["cached"] >= 1, counts

    df = load_cached("kenya_2022")
    assert len(df) > 0, "load_cached returned empty DataFrame after a successful pull"
    assert {"GKGRECORDID", "DATE", "SourceCommonName", "election"} <= set(df.columns)


# ---------------------------------------------------------------------------
# Cleaning module smoke tests
# ---------------------------------------------------------------------------

def _toy_df():
    """Tiny synthetic GKG-shaped DataFrame for offline cleaning smoke tests."""
    import pandas as pd
    return pd.DataFrame([
        {  # African outlet, Kenya election
            "GKGRECORDID": "a",
            "DATE": pd.Timestamp("2022-08-09T12:00:00+00:00"),
            "SourceCommonName": "nation.africa",
            "DocumentIdentifier": "https://nation.africa/kenya/news/election-results",
            "V1Themes": "ELECTION;DEMOCRACY",
            "V2EnhancedThemes": "ELECTION,0;DEMOCRACY,10",
            "V1Locations": "",
            "V21AllNames": "William Ruto,0;Raila Odinga,30",
            "V2Tone": "",
            "election": "kenya_2022",
        },
        {  # International outlet, US politics — should NOT pass relevance
            "GKGRECORDID": "b",
            "DATE": pd.Timestamp("2022-08-09T13:00:00+00:00"),
            "SourceCommonName": "yahoo.com",
            "DocumentIdentifier": "https://yahoo.com/sports/baseball-trade",
            "V1Themes": "SPORTS",
            "V2EnhancedThemes": "SPORTS,0",
            "V1Locations": "",
            "V21AllNames": "Some Player,0",
            "V2Tone": "",
            "election": "kenya_2022",
        },
        {  # BBC Africa URL — should be Edge_BBC_Africa
            "GKGRECORDID": "c",
            "DATE": pd.Timestamp("2022-08-09T14:00:00+00:00"),
            "SourceCommonName": "bbc.com",
            "DocumentIdentifier": "https://www.bbc.com/news/world-africa-12345",
            "V1Themes": "ELECTION",
            "V2EnhancedThemes": "ELECTION,0",
            "V1Locations": "",
            "V21AllNames": "Kenya,0",
            "V2Tone": "",
            "election": "kenya_2022",
        },
    ])


def test_attribute_outlet_origin_basic():
    from elections_frames.cleaning import attribute_outlet_origin

    out = attribute_outlet_origin(_toy_df())
    by_id = dict(zip(out["GKGRECORDID"], out["outlet_origin"]))
    assert by_id["a"] == "African"
    assert by_id["b"] == "International"
    assert by_id["c"] == "Edge_BBC_Africa"


def test_filter_relevant_word_boundary():
    """`Ba` (Senegal candidate) must not match every word with `ba` in it."""
    import pandas as pd

    from elections_frames.cleaning import filter_relevant
    from elections_frames.data import ELECTIONS

    df = pd.DataFrame([
        {  # Real Senegal candidate "Amadou Ba" — should match
            "DocumentIdentifier": "",
            "V21AllNames": "Amadou Ba,0",
            "V2EnhancedThemes": "ELECTION,0",
            "V1Themes": "ELECTION",
            "SourceCommonName": "seneweb.com",
        },
        {  # "baseball" contains "ba" but is NOT election content
            "DocumentIdentifier": "https://yahoo.com/sports/baseball-trade",
            "V21AllNames": "Some Player,0",
            "V2EnhancedThemes": "SPORTS,0",
            "V1Themes": "SPORTS",
            "SourceCommonName": "yahoo.com",
        },
    ])
    kept, _ = filter_relevant(df, ELECTIONS["senegal_2024"], method="hybrid")
    assert len(kept) == 1
    assert kept.iloc[0]["SourceCommonName"] == "seneweb.com"


def test_canonicalize_url_strips_tracking():
    from elections_frames.cleaning import canonicalize_url

    assert canonicalize_url("https://www.bbc.com/news/world-africa-12345/") == "bbc.com/news/world-africa-12345"
    assert canonicalize_url(
        "https://news.example.com/article?utm_source=twitter&id=42&utm_campaign=foo#section"
    ) == "news.example.com/article?id=42"


def test_deduplicate_collapses_syndication():
    """Three rows with identical metadata snippet should collapse to one."""
    import pandas as pd

    from elections_frames.cleaning import deduplicate

    rows = [
        {
            "DocumentIdentifier": f"https://outlet{i}.example.com/article",
            "text_snippet": "Title: same story | Names: William Ruto, Raila Odinga | Themes: ELECTION, DEMOCRACY",
            "DATE": pd.Timestamp("2022-08-09T12:00:00+00:00"),
        }
        for i in range(3)
    ]
    rows.append({  # distinct snippet
        "DocumentIdentifier": "https://other.example.com/article",
        "text_snippet": "Title: different story | Names: someone else | Themes: SPORTS",
        "DATE": pd.Timestamp("2022-08-09T13:00:00+00:00"),
    })
    df = pd.DataFrame(rows)
    deduped, counts = deduplicate(df, threshold=0.8)
    assert counts["n_text_duplicates"] == 2
    assert len(deduped) == 2


def test_sample_eval_candidates_returns_labeled_schema():
    """The candidates DataFrame must carry the columns the labeling UI needs."""
    import pandas as pd

    from elections_frames.cleaning import sample_eval_candidates

    vote_day = pd.Timestamp("2022-08-09T00:00:00+00:00")
    df = pd.DataFrame([
        {
            "GKGRECORDID": f"r{i}",
            "DATE": vote_day + pd.Timedelta(days=(i % 60) - 30),
            "SourceCommonName": "nation.africa" if i % 2 == 0 else "yahoo.com",
            "DocumentIdentifier": f"https://example.com/{i}",
            "election": "kenya_2022",
            "outlet_origin": "African" if i % 2 == 0 else "International",
            "text_snippet": "Title: example",
        }
        for i in range(120)
    ])
    sampled = sample_eval_candidates(df, target_total=40)
    required = {"frame_labels", "labeler_notes", "too_thin", "labeled", "labeled_at"}
    assert required <= set(sampled.columns)
    assert all(v == [] for v in sampled["frame_labels"])
    assert sampled["labeled"].sum() == 0
