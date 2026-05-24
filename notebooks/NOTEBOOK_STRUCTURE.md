# Notebook structure (Project 5: African elections frames)

This notebook reads top-to-bottom **as a lab notebook / short paper**, not as a
code dump. The audience is a recruiter or fellow analyst who wants to see
*how decisions were made*, not just what they were.

The portfolio's brand is **balance practicality vs. perfectionism**. Every
meaningful data-processing decision is *educated by analysis*, not chosen by
convention — the "Show Your Work" discipline.

## The decision discipline (core)

**Every meaningful data-processing decision** — cleaning, imputing,
transforming, categorizing, feature engineering, outlier handling, threshold
setting, aggregation, deduplication — must be presented in **five parts**:

1. **Problem / choice point.** What needs to be decided and why it matters
   for the analysis downstream.
2. **Diagnostic analysis.** Actual code that explores the data to inform the
   choice — distributions, missingness patterns, comparisons across candidate
   options. Use the helpers in `src/elections_frames/diagnostics.py`
   (`missingness_summary`, `missingness_pattern`, `distribution_summary`,
   `distribution_compare`, `before_after`, `compare_alternatives`).
3. **Options considered.** At least 2–3 reasonable alternatives, named explicitly.
4. **Decision + rationale.** What we chose, and why this option, anchored in
   the diagnostic finding above (not in convention).
5. **Sensitivity** (where applicable). How much does the final result depend
   on this choice? Show a robustness check (or point at `03_robustness.ipynb`),
   or explicitly note that the question is not sensitive to this choice.

This block belongs **inline in the notebook**, in markdown, immediately
preceding the code that implements the decision.

### Decision-block template

Copy-paste this as a markdown cell at every choice point:

```markdown
### Decision: <one-line name of the decision>

**Problem.** <Why is this a choice? What goes wrong if we pick badly?>

**Diagnostic.** <One sentence framing.>
```

```python
# Code cell — actual diagnostic analysis using diagnostics.py helpers
from elections_frames.diagnostics import missingness_summary
missingness_summary(df)
```

```markdown
**Options considered.**
- (a) <Option A — one line>
- (b) <Option B — one line>
- (c) <Option C — one line>

**Decision.** <The option we picked.> <One-sentence rationale anchored in the diagnostic.>

**Sensitivity.** <Either: result was robust to alternative X (see `03_robustness.ipynb`).
Or: not sensitive — choice affects only intermediate counts, not the headline finding.
Or: N/A — this is a definitional choice with no quantitative outcome.>
```

### What counts as a "meaningful" decision for this project

Yes, document (the major expected decisions):

- Outlet origin attribution (African / International / Edge — and the edge-case verdicts)
- Article-relevance filter (theme-tag, keyword, or hybrid; threshold)
- Deduplication strategy (URL canonicalization + MinHash threshold)
- Frame taxonomy granularity (5, 6, or 7 frames — informed by labeled boundary cases)
- Prompt iteration (why each version exists; per-version eval scores)
- Confidence threshold for accepting LLM labels (precision vs. coverage curve)
- Eval set sampling strategy (stratification choices)

No, don't document:

- Mechanical type conversions, trivial renames, display formatting

When in doubt: document.

## Notebook layout for this project

- `02_main.ipynb` — THE deliverable. Sections: question, data, ingestion
  status, **outlet attribution decision block**, **relevance filter decision
  block**, **dedup decision block**, structured dataset, classification (calls
  classifier; references `04_pipeline_eval.ipynb`), analysis (frame mix by
  outlet origin, by election, around vote day), hero figure, findings,
  limitations, decisions summary table, reproducibility.
- `03_robustness.ipynb` — sensitivity checks (taxonomy granularity,
  confidence threshold, dedup threshold).
- `04_pipeline_eval.ipynb` — eval loop, per-prompt-version score table,
  confusion matrices, qualitative error analysis, **frame taxonomy
  granularity decision block** (often lives here, not in `02_main`).
- `_scratch/01_explore.ipynb` — ugly EDA (gitignored).

## Output discipline

- Keep notebook outputs **committed** — they are part of the GitHub deliverable.
- Strip large/binary outputs (> 200 KB) by saving to `figures/` and
  referencing with `![](figures/foo.png)` instead.
- Clear cell-execution-counts before final commit (`Kernel → Restart & Run All`,
  then save).
