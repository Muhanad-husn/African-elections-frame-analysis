"""Eval scoring for the frame classifier.

Mechanical scoring helpers, kept in ``src/`` (not the notebook) because they are
reusable across prompt versions, the threshold curve, and the
robustness checks — and because they are unit-testable. The *narrative* (which
prompt version, why each fix) lives in ``notebooks/04_pipeline_eval.ipynb``.

The frame task is **multi-label with abstention**: each article carries a set of
foregrounded frames, and the empty set means "no frame is clearly foregrounded"
(how 135/250 eval rows were hand-labeled, because the GKG-metadata-only snippet
is too thin). So the headline metric is per-frame binary precision/recall/F1 over
the six frames (``n=250``; an empty gold label contributes only potential false
positives), plus an explicit **abstain** precision/recall — did the model
correctly decline to frame a thin snippet?

Public surface:

- ``to_frame_set`` — normalize a label cell (list / ndarray / None) to a frozenset.
- ``per_frame_scores`` — per-frame + abstain + micro/macro table for one run.
- ``run_summary`` — scalar headline metrics for one run (for the per-version table).
- ``confusion_matrix_primary`` — single-label confusion (6 frames + ``none``).
- ``compare_versions`` — stack ``run_summary`` rows across prompt versions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

FRAMES: tuple[str, ...] = (
    "security",
    "economy",
    "democracy",
    "identity",
    "process",
    "corruption",
)
NONE_LABEL = "none"  # confusion-matrix sentinel for abstain / no-frame


def to_frame_set(labels: object) -> frozenset[str]:
    """Normalize a label cell to a frozenset of frame strings.

    Accepts a Python list, a numpy array (how parquet round-trips a list cell),
    ``None``/NaN, or a single string. Unknown tokens are kept as-is so a typo in
    the gold data surfaces loudly rather than being silently dropped.
    """
    if labels is None:
        return frozenset()
    if isinstance(labels, float) and np.isnan(labels):
        return frozenset()
    if isinstance(labels, str):
        labels = [labels]
    return frozenset(str(x) for x in labels)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def per_frame_scores(
    gold: Sequence[Iterable[str]],
    pred: Sequence[Iterable[str]],
    frames: Sequence[str] = FRAMES,
) -> pd.DataFrame:
    """Per-frame binary P/R/F1 over ``frames``, plus abstain and averages.

    ``gold`` / ``pred`` are parallel sequences of frame-label iterables (an empty
    iterable = abstain / no frame). Returns a DataFrame indexed by frame with
    ``support`` (gold positives), ``pred`` (predicted positives), ``tp/fp/fn``,
    and ``precision/recall/f1``. Trailing rows: ``abstain`` (the no-frame decision
    scored as its own class), ``micro avg`` and ``macro avg`` over ``frames``.

    ``frames`` defaults to the six-frame taxonomy but can be any label set — e.g.
    a merged 5-frame taxonomy whose labels include a super-label like
    ``"governance"``. It MUST cover every label that appears in ``gold``/``pred``,
    otherwise rows relabeled to an out-of-set label would be silently dropped and
    the micro/macro averages would be measuring deletion rather than the merge.
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    frames = tuple(frames)
    gold_sets = [to_frame_set(g) for g in gold]
    pred_sets = [to_frame_set(p) for p in pred]
    seen = {f for s in (*gold_sets, *pred_sets) for f in s}
    missing = seen - set(frames)
    if missing:
        raise ValueError(f"labels not in `frames` would be dropped: {sorted(missing)}")

    rows: dict[str, dict[str, float]] = {}
    for f in frames:
        tp = sum(f in g and f in p for g, p in zip(gold_sets, pred_sets, strict=True))
        fp = sum(f not in g and f in p for g, p in zip(gold_sets, pred_sets, strict=True))
        fn = sum(f in g and f not in p for g, p in zip(gold_sets, pred_sets, strict=True))
        precision, recall, f1 = _prf(tp, fp, fn)
        rows[f] = {
            "support": sum(f in g for g in gold_sets),
            "pred": sum(f in p for p in pred_sets),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    # Abstain scored as its own class: positive == empty label.
    a_tp = sum(not g and not p for g, p in zip(gold_sets, pred_sets, strict=True))
    a_fp = sum(bool(g) and not p for g, p in zip(gold_sets, pred_sets, strict=True))
    a_fn = sum(not g and bool(p) for g, p in zip(gold_sets, pred_sets, strict=True))
    a_p, a_r, a_f1 = _prf(a_tp, a_fp, a_fn)
    rows["abstain"] = {
        "support": sum(not g for g in gold_sets),
        "pred": sum(not p for p in pred_sets),
        "tp": a_tp, "fp": a_fp, "fn": a_fn,
        "precision": a_p, "recall": a_r, "f1": a_f1,
    }

    df = pd.DataFrame(rows).T

    frame_rows = df.loc[list(frames)]
    micro_tp, micro_fp, micro_fn = (frame_rows[c].sum() for c in ("tp", "fp", "fn"))
    mi_p, mi_r, mi_f1 = _prf(int(micro_tp), int(micro_fp), int(micro_fn))
    df.loc["micro avg"] = {
        "support": frame_rows["support"].sum(), "pred": frame_rows["pred"].sum(),
        "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
        "precision": mi_p, "recall": mi_r, "f1": mi_f1,
    }
    df.loc["macro avg"] = {
        "support": frame_rows["support"].sum(), "pred": frame_rows["pred"].sum(),
        "tp": np.nan, "fp": np.nan, "fn": np.nan,
        "precision": frame_rows["precision"].mean(),
        "recall": frame_rows["recall"].mean(),
        "f1": frame_rows["f1"].mean(),
    }
    int_cols = ["support", "pred", "tp", "fp", "fn"]
    df[int_cols] = df[int_cols].apply(pd.to_numeric)
    return df


def exact_match_ratio(
    gold: Sequence[Iterable[str]], pred: Sequence[Iterable[str]]
) -> float:
    """Subset accuracy: fraction of rows whose predicted set equals the gold set."""
    gs = [to_frame_set(g) for g in gold]
    ps = [to_frame_set(p) for p in pred]
    return float(np.mean([g == p for g, p in zip(gs, ps, strict=True)])) if gs else 0.0


def run_summary(
    gold: Sequence[Iterable[str]],
    pred: Sequence[Iterable[str]],
    version: str | None = None,
    frames: Sequence[str] = FRAMES,
) -> dict[str, float | str]:
    """Scalar headline metrics for one run — one row of the per-version table."""
    table = per_frame_scores(gold, pred, frames=frames)
    out: dict[str, float | str] = {
        "micro_f1": table.loc["micro avg", "f1"],
        "macro_f1": table.loc["macro avg", "f1"],
        "micro_precision": table.loc["micro avg", "precision"],
        "micro_recall": table.loc["micro avg", "recall"],
        "abstain_precision": table.loc["abstain", "precision"],
        "abstain_recall": table.loc["abstain", "recall"],
        "abstain_f1": table.loc["abstain", "f1"],
        "exact_match": exact_match_ratio(gold, pred),
        "n": len(gold),
    }
    if version is not None:
        out = {"version": version, **out}
    return out


def _is_missing(x: object) -> bool:
    """True for ``None``, NaN, or empty string — all mean 'abstain' for a primary."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return x == ""


def primary_of(labels: Iterable[str], explicit_primary: str | None = None) -> str:
    """Reduce a label set to a single confusion-matrix label.

    Uses ``explicit_primary`` when given (the model reports one); otherwise the
    sole frame, or ``NONE_LABEL`` for abstain. A multi-label set with no explicit
    primary is reduced to its first element (callers restrict the confusion matrix
    to single-or-empty gold rows, so this fallback is for predictions only).
    """
    s = list(to_frame_set(labels))
    if not _is_missing(explicit_primary):
        return str(explicit_primary)
    if not s:
        return NONE_LABEL
    return s[0]


def confusion_matrix_primary(
    gold: Sequence[Iterable[str]],
    pred_primary: Sequence[str | None],
    restrict_single_gold: bool = True,
) -> pd.DataFrame:
    """Single-label confusion matrix over 6 frames + ``none`` (abstain).

    Rows = gold, columns = predicted. The confusion matrix is a single-label tool,
    so by default it is computed on the rows whose **gold** label has at most one
    frame (the bulk: abstain + single-frame rows). Multi-frame gold rows are
    excluded here and analyzed separately in the notebook's qualitative section.

    ``pred_primary`` is the model's stated primary frame (``None``/NaN when the
    model abstained, which maps to the ``none`` column).
    """
    labels = [*FRAMES, NONE_LABEL]
    cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for g, pp in zip(gold, pred_primary, strict=True):
        gset = to_frame_set(g)
        if restrict_single_gold and len(gset) > 1:
            continue
        g_label = next(iter(gset)) if gset else NONE_LABEL
        p_label = NONE_LABEL if _is_missing(pp) else str(pp)
        cm.loc[g_label, p_label] += 1
    cm.index.name = "gold"
    cm.columns.name = "pred"
    return cm


def compare_versions(summaries: Iterable[dict[str, float | str]]) -> pd.DataFrame:
    """Stack ``run_summary`` dicts into a per-version comparison table."""
    df = pd.DataFrame(list(summaries))
    if "version" in df.columns:
        df = df.set_index("version")
    return df
