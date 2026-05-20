"""Visualization helpers — matplotlib + seaborn.

Style chosen for static, thumbnail-legible figures (LinkedIn/Twitter previews,
recruiter-facing notebook scrubs). Stacked bars and confusion matrices live or
die on legible static rendering, not interactivity. See ``README.md`` Visual
style section.

Filled in during Session 7 (see ``IMPLEMENTATION_PLAN.md``). Planned surface:

- ``stacked_frame_bar(df, by="outlet_origin")`` — the headline figure shape.
- ``confusion_matrix_plot(...)`` — used in ``04_pipeline_eval.ipynb``.
- ``per_election_panel(...)`` — small-multiples by election.
- ``save_figure(fig, name)`` — wrapped ``savefig`` to ``figures/<name>.png``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def save_figure(fig, name: str, dpi: int = 200) -> Path:
    """Save a matplotlib Figure to ``figures/<name>.png``."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
