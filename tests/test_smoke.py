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
    """All planned modules import (most are stubs until later sessions)."""
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
    """The six diagnostic helpers named in CLAUDE.md are importable."""
    from elections_frames.diagnostics import (  # noqa: F401
        before_after,
        compare_alternatives,
        distribution_compare,
        distribution_summary,
        missingness_pattern,
        missingness_summary,
    )


def test_prompts_package_present():
    """Versioned prompts package is importable (versions added in Session 4+)."""
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
