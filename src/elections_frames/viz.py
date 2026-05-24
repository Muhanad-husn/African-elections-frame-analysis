"""Visualization helpers — matplotlib + seaborn.

Style chosen for static, thumbnail-legible figures (LinkedIn/Twitter previews,
recruiter-facing notebook scrubs). Stacked bars and confusion matrices live or
die on legible static rendering, not interactivity. See ``README.md`` Visual
style section.

Public surface (built in Session 7):

- ``set_style()`` — apply the project-wide matplotlib/seaborn theme.
- ``frame_share_matrix(df, by)`` — aggregate primary-frame mix into a
  group x frame share table (the input shape every chart consumes).
- ``stacked_frame_bar(df, by="outlet_origin")`` — the headline figure shape.
- ``per_election_panel(df, by="outlet_origin")`` — small-multiples by election
  (this is the hero figure shape).
- ``frame_share_timeline(df, election)`` — frame mix over days around vote day.
- ``confusion_matrix_plot(cm, labels)`` — used in ``04_pipeline_eval.ipynb``.
- ``save_figure(fig, name)`` — wrapped ``savefig`` to ``figures/<name>.png``.

Design note: aggregation always runs over **framed** rows (``pred_primary``
non-null), i.e. abstentions and hard-error rows are excluded from the mix.
Abstention *rate* is reported separately in the notebook — it is a finding,
not a frame. Shares are computed on the **primary** frame so each article
contributes exactly one unit and bars sum to 100%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Canonical frame order — used for stacking order, legend order, and column
# order everywhere so colours stay stable across every figure in the project.
FRAMES = ["security", "economy", "democracy", "identity", "process", "corruption"]

# Okabe-Ito colourblind-safe qualitative palette, hand-mapped to frames.
# Vermillion reads as "security/danger"; the rest are simply distinct.
FRAME_COLORS = {
    "security": "#D55E00",   # vermillion
    "economy": "#0072B2",    # blue
    "democracy": "#009E73",  # bluish green
    "identity": "#CC79A7",   # reddish purple
    "process": "#E69F00",    # orange
    "corruption": "#56B4E9",  # sky blue
}

# Human-facing labels for outlet-origin codes.
ORIGIN_LABELS = {
    "African": "African press",
    "International": "International press",
}


def set_style() -> None:
    """Apply the project-wide figure theme (call once at the top of a notebook)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 11,
        }
    )


def framed(df: pd.DataFrame, frame_col: str = "pred_primary") -> pd.DataFrame:
    """Rows where the model foregrounded a frame (excludes abstentions + errors).

    Abstained rows and the 130 hard-error rows both carry a null
    ``pred_primary``, so a single ``notna`` filter drops both.
    """
    return df[df[frame_col].notna()]


def frame_share_matrix(
    df: pd.DataFrame,
    by: str | list[str],
    frame_col: str = "pred_primary",
    frames: list[str] = FRAMES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate the primary-frame mix per group.

    Returns ``(shares, sizes)`` where ``shares`` is a group x frame table whose
    rows each sum to 1.0 (frame columns in canonical order) and ``sizes`` is the
    number of *framed* articles per group. Aggregation runs over framed rows
    only — see module docstring.
    """
    sub = framed(df, frame_col)
    counts = (
        sub.groupby(by)[frame_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=frames, fill_value=0)
    )
    sizes = counts.sum(axis=1)
    shares = counts.div(sizes, axis=0)
    return shares, sizes


def _stack_on_ax(
    ax: plt.Axes,
    shares: pd.DataFrame,
    sizes: pd.Series | None = None,
    frames: list[str] = FRAMES,
    annotate_sizes: bool = True,
    xtick_labels: dict | None = None,
) -> None:
    """Draw a vertical stacked bar of frame shares onto ``ax`` (internal)."""
    groups = list(shares.index)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for frame in frames:
        vals = shares[frame].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=FRAME_COLORS[frame], label=frame,
               width=0.72, edgecolor="white", linewidth=0.6)
        bottom += vals

    labels = [str(g) for g in groups]
    if xtick_labels:
        labels = [xtick_labels.get(g, str(g)) for g in groups]
    if annotate_sizes and sizes is not None:
        labels = [f"{lab}\n(n={int(sizes[g]):,})" for lab, g in zip(labels, groups, strict=True)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.margins(x=0.08)
    ax.grid(axis="x", visible=False)


def stacked_frame_bar(
    df: pd.DataFrame,
    by: str = "outlet_origin",
    frame_col: str = "pred_primary",
    frames: list[str] = FRAMES,
    title: str | None = None,
    ax: plt.Axes | None = None,
    annotate_sizes: bool = True,
    figsize: tuple[float, float] = (7, 6),
) -> plt.Figure:
    """Stacked bar of primary-frame mix, one bar per ``by`` group.

    The headline-figure shape: pass ``by="outlet_origin"`` on a single
    election's rows for an African-vs-International comparison.
    """
    shares, sizes = frame_share_matrix(df, by, frame_col, frames)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    _stack_on_ax(ax, shares, sizes, frames, annotate_sizes,
                 xtick_labels=ORIGIN_LABELS if by == "outlet_origin" else None)
    ax.set_ylabel("share of framed articles")
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title="frame", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def per_election_panel(
    df: pd.DataFrame,
    by: str = "outlet_origin",
    elections: list[str] | None = None,
    election_titles: dict | None = None,
    frame_col: str = "pred_primary",
    frames: list[str] = FRAMES,
    suptitle: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    annotate_sizes: bool = True,
) -> plt.Figure:
    """Small-multiples: one stacked-bar subplot per election (the hero shape).

    ``elections`` controls panel order; defaults to the order they appear in
    ``df``. ``election_titles`` maps election keys to display titles.
    """
    if elections is None:
        elections = list(pd.unique(df["election"]))
    n = len(elections)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax, election in zip(axes_flat, elections, strict=False):
        sub = df[df["election"] == election]
        shares, sizes = frame_share_matrix(sub, by, frame_col, frames)
        _stack_on_ax(ax, shares, sizes, frames, annotate_sizes,
                     xtick_labels=ORIGIN_LABELS if by == "outlet_origin" else None)
        title = election_titles.get(election, election) if election_titles else election
        ax.set_title(title, fontsize=12)
    for ax in axes_flat[n:]:
        ax.axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel("share of framed articles")

    handles = [plt.Rectangle((0, 0), 1, 1, color=FRAME_COLORS[f]) for f in frames]
    fig.legend(handles, frames, title="frame", loc="lower center",
               ncol=len(frames), bbox_to_anchor=(0.5, -0.02), frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97 if suptitle else 1.0))
    return fig


def frame_share_timeline(
    df: pd.DataFrame,
    election: str,
    vote_day: pd.Timestamp,
    by: str = "outlet_origin",
    frame_col: str = "pred_primary",
    frames: list[str] = FRAMES,
    rolling: int = 7,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Frame-share lines over days relative to vote day, faceted by ``by`` group.

    Each line is a frame's share of framed articles per day, smoothed with a
    centred ``rolling``-day mean. Day 0 is vote day (dashed vertical rule).
    """
    sub = framed(df[df["election"] == election], frame_col).copy()
    sub["rel_day"] = (sub["DATE"].dt.tz_convert("UTC").dt.floor("D")
                      - vote_day.tz_convert("UTC").floor("D")).dt.days

    groups = list(pd.unique(sub[by]))
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True, squeeze=False)
    for ax, group in zip(axes[0], groups, strict=True):
        g = sub[sub[by] == group]
        daily = (g.groupby("rel_day")[frame_col].value_counts()
                 .unstack(fill_value=0).reindex(columns=frames, fill_value=0))
        daily = daily.reindex(range(daily.index.min(), daily.index.max() + 1), fill_value=0)
        share = daily.div(daily.sum(axis=1).replace(0, np.nan), axis=0)
        smooth = share.rolling(rolling, center=True, min_periods=max(1, rolling // 2)).mean()
        for frame in frames:
            ax.plot(smooth.index, smooth[frame], color=FRAME_COLORS[frame],
                    label=frame, linewidth=2)
        ax.axvline(0, color="0.3", linestyle="--", linewidth=1)
        ax.set_title(ORIGIN_LABELS.get(group, str(group)) if by == "outlet_origin" else str(group))
        ax.set_xlabel("days from vote day")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0][0].set_ylabel(f"frame share ({rolling}-day mean)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="frame", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(
    cm: pd.DataFrame | np.ndarray,
    labels: list[str] | None = None,
    normalize: bool = False,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
    cmap: str = "rocket_r",
) -> plt.Figure:
    """Heatmap of a confusion matrix (gold rows x predicted columns).

    Accepts a labelled ``DataFrame`` or a raw array + ``labels``. With
    ``normalize=True`` each gold row is shown as a fraction summing to 1.
    """
    if not isinstance(cm, pd.DataFrame):
        cm = pd.DataFrame(cm, index=labels, columns=labels)
    if normalize:
        mat = cm.astype(float).div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        fmt = ".2f"
    else:
        mat = cm.astype(int)
        fmt = "d"

    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                cbar_kws={"label": "row share" if normalize else "count"},
                linewidths=0.5, linecolor="white", square=True)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold (hand label)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig, name: str, dpi: int = 200) -> Path:
    """Save a matplotlib Figure to ``figures/<name>.png``."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
