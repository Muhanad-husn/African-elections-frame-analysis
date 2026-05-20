"""Smoke tests — verify the package imports and data paths exist.

These exist from Session 1 onward. Real functional smoke tests (e.g. a
1-day GDELT pull, a mocked classify_article call) are added in Sessions 2
and 4.
"""


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
