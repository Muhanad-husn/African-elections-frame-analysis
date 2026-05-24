"""Unit tests for the eval scoring module."""

from __future__ import annotations

import numpy as np


def test_to_frame_set_normalizes_inputs():
    from elections_frames.eval import to_frame_set

    assert to_frame_set(["security", "process"]) == frozenset({"security", "process"})
    assert to_frame_set(np.array(["economy"], dtype=object)) == frozenset({"economy"})
    assert to_frame_set([]) == frozenset()
    assert to_frame_set(None) == frozenset()
    assert to_frame_set(float("nan")) == frozenset()
    assert to_frame_set("identity") == frozenset({"identity"})


def test_per_frame_scores_perfect():
    """Identical gold and pred -> precision=recall=f1=1 on the present frames."""
    from elections_frames.eval import per_frame_scores

    gold = [["security"], ["economy", "democracy"], []]
    pred = [["security"], ["economy", "democracy"], []]
    t = per_frame_scores(gold, pred)
    assert t.loc["security", "f1"] == 1.0
    assert t.loc["economy", "f1"] == 1.0
    assert t.loc["micro avg", "f1"] == 1.0
    # Abstain: one empty gold correctly predicted empty.
    assert t.loc["abstain", "tp"] == 1
    assert t.loc["abstain", "f1"] == 1.0


def test_per_frame_scores_counts_fp_fn():
    from elections_frames.eval import per_frame_scores

    # gold says abstain; model says process -> process FP, abstain FN.
    gold = [[], ["security"]]
    pred = [["process"], ["process"]]
    t = per_frame_scores(gold, pred)
    assert t.loc["process", "fp"] == 2  # process predicted on both, gold has it nowhere
    assert t.loc["process", "tp"] == 0
    assert t.loc["security", "fn"] == 1  # missed the real security label
    assert t.loc["abstain", "support"] == 1
    assert t.loc["abstain", "recall"] == 0.0  # gold-abstain row was not predicted abstain


def test_per_frame_scores_custom_frames_merge():
    """Scoring over a merged taxonomy counts the super-label (not silently dropped)."""
    import pytest

    from elections_frames.eval import per_frame_scores

    # gold=process, pred=democracy: a dem<->proc confusion. Merged, it becomes correct.
    gold = [["process"], ["democracy"]]
    pred = [["democracy"], ["democracy"]]
    merged_frames = ["security", "economy", "identity", "corruption", "governance"]
    g = [["governance"], ["governance"]]
    p = [["governance"], ["governance"]]
    t = per_frame_scores(g, p, frames=merged_frames)
    assert t.loc["governance", "f1"] == 1.0
    assert t.loc["micro avg", "f1"] == 1.0

    # A label outside `frames` must raise rather than be dropped.
    with pytest.raises(ValueError, match="not in"):
        per_frame_scores(gold, pred, frames=["security", "economy"])


def test_run_summary_keys():
    from elections_frames.eval import run_summary

    s = run_summary([["security"], []], [["security"], []], version="vX")
    assert s["version"] == "vX"
    assert s["n"] == 2
    assert s["micro_f1"] == 1.0
    assert s["abstain_f1"] == 1.0
    assert s["exact_match"] == 1.0


def test_confusion_matrix_excludes_multi_gold_and_counts_abstain():
    from elections_frames.eval import NONE_LABEL, confusion_matrix_primary

    gold = [["security"], [], ["economy", "process"]]  # last is multi -> excluded
    pred_primary = ["process", None, "economy"]
    cm = confusion_matrix_primary(gold, pred_primary)
    assert cm.loc["security", "process"] == 1  # security misframed as process
    assert cm.loc[NONE_LABEL, NONE_LABEL] == 1  # abstain correctly abstained
    assert cm.values.sum() == 2  # the multi-gold row was excluded


def test_compare_versions_indexes_by_version():
    from elections_frames.eval import compare_versions, run_summary

    s1 = run_summary([["security"]], [["security"]], version="v1")
    s2 = run_summary([["security"]], [[]], version="v2")
    table = compare_versions([s1, s2])
    assert list(table.index) == ["v1", "v2"]
    assert table.loc["v1", "micro_f1"] == 1.0
    assert table.loc["v2", "micro_f1"] == 0.0
