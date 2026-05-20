# African Elections Frame Analysis — Multi-Session Implementation Plan

> **Created:** 2026-05-20
> **Source plan:** Synthesized from `CLAUDE.md` (Done definition + Expected major decisions) and `README.md`. No standalone plan file existed prior to this.
> **Total sessions:** 9
> **Estimated total effort:** ~9 sessions, with a hard external dependency at Session 3→5 (Muhanad hand-labels the eval set between those sessions; Session 4 can be done in parallel with that labeling).

## Overview

End-to-end build of a four-stage pipeline (ingest → clean → classify → evaluate) for measuring frame divergence between African and international press coverage of four African elections (Nigeria 2023, Kenya 2022, Senegal 2024, South Africa 2024). The flagship deliverable is the **pipeline + eval + analysis modularity** with every data-processing decision documented in five-part form inline in the notebooks.

## API choice (resolved)

The classifier uses **NVIDIA NIM** (`deepseek-ai/deepseek-v4-pro` primary, `minimaxai/minimax-m2.7` hard-failure fallback) via the OpenAI-compatible client at `https://integrate.api.nvidia.com/v1`. API key lives in `../secrets.toml` under `[NVIDIA] API_KEY`. Call shape reference: `../NVIDIA_API_request_sample.py`. The `CLAUDE.md` reference to "Anthropic API" was stale and has been updated to match.

## Session Dependency Graph

```
1 (scaffold)
 └─> 2 (ingest)
      └─> 3 (clean + eval-set sampling + labeling UI)
           ├─> [EXTERNAL: Muhanad hand-labels eval set]
           │    └─> 5 (eval loop + prompt iteration)
           │         └─> 6 (threshold + production run)
           │              └─> 7 (analysis notebook + hero figure)
           │                   └─> 8 (robustness)
           │                        └─> 9 (README polish + verification)
           └─> 4 (classifier module + prompt v1)  [parallel with labeling]
                └─> 5
```

Session 4 has no dependency on the labeled eval set and should be done in parallel with Muhanad's labeling work to minimize wall-clock time.

---

## Session 1: Project scaffolding & setup

**Objective:** Stand up the project structure (directories, `pyproject.toml`, `src/` skeleton, notebook scaffolding, `NOTEBOOK_STRUCTURE.md`, `codebook.md` v0, `outlets.csv` skeleton, secrets handling, smoke test scaffold) so all subsequent sessions can write into a coherent layout.
**Inputs:** `CLAUDE.md`, `README.md`, `../secrets.toml` (for key shape), `../NVIDIA_API_request_sample.py` (for call shape).
**Outputs:**
- `pyproject.toml` with `[viz, nlp, llm]` extras declared
- `src/elections_frames/` package skeleton (empty `data.py`, `cleaning.py`, `classify.py`, `viz.py`, `diagnostics.py`, `prompts/__init__.py`)
- `notebooks/` with `02_main.ipynb`, `03_robustness.ipynb`, `04_pipeline_eval.ipynb` (skeleton headers only), `NOTEBOOK_STRUCTURE.md` (five-part block template), `_scratch/.gitkeep`
- `data/external/outlets.csv` with header + edge-case columns (BBC Africa, Al Jazeera English Africa desk, Reuters Africa, RFI Afrique — verdicts blank for now)
- `data/external/codebook.md` — v0 of the 6-frame taxonomy (security / economy / democracy / identity / process / corruption) with one-sentence definitions
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `figures/.gitkeep`
- `tests/test_smoke.py` scaffold (empty test functions)
- `.gitignore` covering `data/raw/`, `data/processed/*.parquet`, `notebooks/_scratch/`, `__pycache__`, `.venv`
**Depends on:** None.

### Context for resumption

This is Session 1 of 9. Nothing exists yet except `CLAUDE.md`, `README.md`, and `IMPLEMENTATION_PLAN.md`. Read all three in full before scaffolding. The project's brand is "neat lab work, not ship-ready apps" — keep the package code lean and push methodological rigor into the notebooks. The classifier API is locked to NVIDIA NIM (OpenAI-compatible, `deepseek-v4-pro` primary, `minimax-m2.7` fallback) — see "API choice (resolved)" above.

### Steps

1. Create directory tree as listed in `CLAUDE.md` "Files orientation" section.
2. Write `pyproject.toml` with project metadata, deps grouped under `viz` (matplotlib, seaborn), `nlp` (datasketch for MinHash, langdetect or fastText for English filter), `llm` (openai client — NVIDIA NIM is OpenAI-compatible, plus `tomli` or `tomllib` for reading `../secrets.toml`). Pin Python ≥3.11.
3. Write `NOTEBOOK_STRUCTURE.md` — the five-part block template (problem / diagnostic / options / decision + rationale / sensitivity) plus a short example. Reference: `_template/notebooks/NOTEBOOK_STRUCTURE.md` in monorepo if present.
4. Write `data/external/codebook.md` v0 — 6 frames with one-sentence definitions each. Note that the taxonomy will be revisited after first batch of labels (per `CLAUDE.md` workflow note).
5. Create `data/external/outlets.csv` with columns: `outlet, domain, origin (African|International|Edge), edge_case_notes, verdict_rule`. Pre-populate the four named edge cases with blank verdicts.
6. Write notebook skeletons with section headers matching the README's planned decisions.
7. Empty smoke tests for each module (`test_data_smoke`, `test_cleaning_smoke`, `test_classify_smoke`).
8. `.gitignore` and a one-line README install instruction sanity check.

### Completion criteria

- [ ] `pip install -e ".[viz,nlp,llm]"` succeeds in a fresh venv
- [ ] `pytest tests/` runs (zero tests pass/fail — just imports succeed)
- [ ] All four notebook skeletons open in JupyterLab without errors
- [ ] Directory tree matches `CLAUDE.md` "Files orientation" exactly

### Handoff notes

*(filled during execution)*

---

## Session 2: GDELT ingestion module

**Objective:** Implement `src/elections_frames/data.py` to pull GDELT 2.0 GKG records for ±30-day windows around each of the 4 elections, with aggressive local caching and a committed manifest of pulled file IDs.
**Inputs:** Scaffolded project from Session 1; election dates from `README.md`; GDELT 2.0 GKG endpoint (http://data.gdeltproject.org/gdeltv2/).
**Outputs:**
- `src/elections_frames/data.py` — `pull_gkg_window(election, days=30)`, `load_cached(election)`, `outlet_provenance_join(df, outlets_csv)`, all with type hints
- `data/raw/<election>/` — cached GKG archives (gitignored)
- `data/raw/manifest.json` — list of file IDs pulled per election (committed)
- Updated `tests/test_smoke.py` — smoke test that hits a 1-day window and asserts non-empty result
- Brief ingestion section appended to `notebooks/02_main.ipynb` showing row counts per election (no decision block yet — just sanity output)
**Depends on:** Session 1.

### Context for resumption

Project scaffolding is complete. The GDELT 2.0 GKG is the only data source for article-level records; pulls are large so cache locally and commit only the manifest. Elections are Nigeria 2023, Kenya 2022, Senegal 2024, South Africa 2024 — election dates are in `README.md` (verify exact dates with public records during this session). The English-relevance filter happens in Session 3, not here — pull everything in the window. Use the `outlets.csv` from Session 1 only as a known-domain list for the provenance column (don't filter on it yet).

### Steps

1. Look up and verify exact election dates for the four elections; document them as constants in `data.py` with citation comments.
2. Implement GKG window puller — iterate 15-minute GKG file slugs across the ±30-day window, download to `data/raw/<election>/`, parse columns of interest (`GKGRECORDID`, `DATE`, `SourceCommonName`, `DocumentIdentifier`, `V2Themes`, `V2Tone`, etc.).
3. Implement caching: idempotent pull (skip files already on disk), append to `manifest.json`.
4. Implement `outlet_provenance_join` — left-join on `SourceCommonName` against `outlets.csv`; unmatched outlets get `origin = "Unknown"` (resolved later).
5. Smoke test: pull a single 24-hour slice for Kenya 2022 and assert row count > 0.
6. Add brief ingestion-status cell to `notebooks/02_main.ipynb` (count of rows per election + count of unknown-origin outlets — this is a teaser for Session 3's outlet-origin decision block).

### Completion criteria

- [ ] All four ±30-day windows pulled and cached
- [ ] `manifest.json` committed with file IDs
- [ ] Smoke test passes
- [ ] Notebook cell shows row counts per election

### Handoff notes

*(filled during execution)*

---

## Session 3: Cleaning module + eval-set candidate sampling + labeling handoff

**Objective:** Implement cleaning (English filter, relevance filter, dedup), implement outlet-origin attribution with edge cases resolved, write three decision blocks in five-part form (outlet attribution, relevance filter, dedup), stratify-sample 200–300 candidate articles for Muhanad to hand-label, and produce a minimal labeling UI/CSV.
**Inputs:** Cached GDELT data from Session 2; `outlets.csv` from Session 1; `codebook.md` v0.
**Outputs:**
- `src/elections_frames/cleaning.py` — `filter_english`, `filter_relevant`, `deduplicate`, all parameterized
- `src/elections_frames/diagnostics.py` — `missingness_summary`, `missingness_pattern`, `distribution_summary`, `distribution_compare`, `before_after`, `compare_alternatives` (the helpers referenced in `CLAUDE.md`)
- `data/external/outlets.csv` — finalized, edge cases resolved with rule + verdict
- `notebooks/02_main.ipynb` — three completed five-part decision blocks (outlet attribution, relevance filter, dedup) + structured dataset preview
- `data/external/eval_set_candidates.parquet` — 200–300 stratified articles (columns: `article_id, date, outlet, outlet_origin, election, text_snippet, frame_labels (BLANK)`)
- `notebooks/eval_labeling.ipynb` OR a simple Streamlit app — labeling UI that loads `eval_set_candidates.parquet`, lets Muhanad pick from the 6 frames, saves to `eval_set.parquet`. Anti-anchoring: never display any LLM-suggested label.
- Brief written handoff (`docs/labeling_handoff.md` or inline in `eval_labeling.ipynb`) explaining the codebook and how to use the UI.
**Depends on:** Session 2.

### Context for resumption

Cached GDELT records are on disk. Outlet provenance has been joined but ~unknown outlets remain. The three decisions in this session — outlet attribution, relevance filter, dedup — are the first methodological-discipline test of the project, so each must follow the five-part block template strictly. **Do not pre-populate eval label suggestions** — `CLAUDE.md` is explicit that the eval set must be hand-labeled blind. After this session, work pauses on the eval-set track until Muhanad finishes labeling. Session 4 (classifier scaffold) can proceed in parallel.

### Steps

1. **Outlet attribution decision block** — diagnose unknown-origin row counts, decide rule for African vs. International, resolve the four named edge cases explicitly (BBC Africa, Al Jazeera English Africa desk, Reuters Africa, RFI Afrique). Document verdicts in `outlets.csv`.
2. Implement `filter_english` (langdetect or fastText) and stage as a callable. Diagnose: what % of rows are non-English, per election.
3. **Relevance filter decision block** — options: theme-tag threshold (V2Themes contains ELECTION_*), keyword filter (election name + candidate names), hybrid. Build a small 30–50 article probe set, eyeball precision/recall on each option, pick one, document.
4. **Dedup decision block** — URL canonicalization first, then MinHash Jaccard at varying thresholds (0.7, 0.8, 0.9), measure false-merge rate on probe set, pick threshold.
5. Write `diagnostics.py` helpers as you use them (don't pre-build).
6. Produce final structured analytic dataset: `data/processed/articles_clean.parquet`.
7. Stratified eval-set sample (200–300 articles): stratify by election × outlet origin × week-around-vote. Sampling code is itself a decision block (sampling strategy = expected decision #7 from CLAUDE.md, but partial — full justification can be revisited in Session 9 once labeled).
8. Labeling UI: simple Jupyter form or Streamlit, frame radio/multiselect, save to `data/external/eval_set.parquet`. No LLM suggestions visible to labeler.
9. Write `docs/labeling_handoff.md` — for Muhanad: 1-page on how to use the UI + a printable codebook reminder.

### Completion criteria

- [ ] Three five-part decision blocks complete in `notebooks/02_main.ipynb`
- [ ] `data/processed/articles_clean.parquet` produced
- [ ] `data/external/eval_set_candidates.parquet` stratified and saved
- [ ] Labeling UI launches and saves a label successfully (verify with one test row, then clear)
- [ ] `outlets.csv` edge cases resolved
- [ ] `docs/labeling_handoff.md` written

### Handoff notes

*(filled during execution — note any candidate articles flagged as ambiguous during sampling, since those will inform Session 5's prompt iteration)*

---

## Session 4: Classifier module + prompt v1 (parallel with eval labeling)

**Objective:** Implement `src/elections_frames/classify.py` (NVIDIA NIM wrapper, OpenAI-compatible client; `deepseek-v4-pro` primary with `minimax-m2.7` hard-failure fallback) with structured output and cost logging; write prompt v1 and store it under `src/elections_frames/prompts/v1.py`; run it on a tiny smoke subset (5–10 articles) to verify the round-trip works. **Do not run on the eval set yet** — that's Session 5, after Muhanad's labels exist.
**Inputs:** Cleaned articles from Session 3; codebook v0; API key from `../secrets.toml`.
**Outputs:**
- `src/elections_frames/classify.py` — `classify_article(text, prompt_version)`, with retry, structured-output parsing, fallback model on hard failure, cost logging
- `src/elections_frames/prompts/v1.py` — prompt v1 as a Python string constant + structured-output schema definition
- `data/processed/llm_cost.csv` — initialized with header, smoke-run row appended
- `tests/test_classify_smoke.py` — mocked + live smoke test (live test gated by env var to avoid CI cost)
**Depends on:** Session 3 (needs cleaned articles to smoke-test on); independent of Muhanad's labeling.

### Context for resumption

This session can be done while Muhanad is hand-labeling the eval set — there's no dependency on labeled data here, only on cleaned articles. API: NVIDIA NIM via OpenAI-compatible client (`deepseek-v4-pro` primary, `minimax-m2.7` fallback on hard failure); call shape in `../NVIDIA_API_request_sample.py`; key in `../secrets.toml`. Cost logging is a first-class requirement (Principle 2). Keep the wrapper small; rigor lives in the prompt iteration (Session 5).

### Steps

1. Read `../NVIDIA_API_request_sample.py` to nail down the call shape and secrets loading.
2. Implement structured output schema — Pydantic or JSON Schema for `{frames: list[Literal["security","economy","democracy","identity","process","corruption"]], confidence: float, rationale: str}`.
3. Implement `classify_article` with: input/output token logging → `llm_cost.csv`, retry on transient errors, hard-failure fallback to the secondary model.
4. Write prompt v1 — codebook-grounded, structured-output instructions, few-shot examples drawn from public articles (NOT from `eval_set.parquet` — don't contaminate the eval set with prompt examples).
5. Smoke test: classify 5–10 cleaned articles (NOT from the eval set), inspect outputs by eye, append cost to log.
6. Mocked unit test in `tests/` for offline CI.

### Completion criteria

- [ ] `classify_article` returns valid structured output for 5–10 articles
- [ ] `llm_cost.csv` has a smoke-run row with non-zero tokens
- [ ] Mocked test passes; live test passes when env var set
- [ ] Prompt v1 saved in `src/elections_frames/prompts/v1.py` with a docstring noting "v1 — initial draft, no eval data yet"

### Handoff notes

*(filled during execution — note any prompt-design surprises that prompt iteration in Session 5 should address)*

---

## Session 5: Eval loop + prompt iteration

**Objective:** With Muhanad's hand-labeled `eval_set.parquet` in hand, build the eval loop (precision/recall/F1 per frame, confusion matrix, qualitative error analysis), then iterate the prompt across versions, committing per-iteration eval scores. The narrative of *why each version exists* is itself a deliverable.
**Inputs:** `data/external/eval_set.parquet` (hand-labeled, from Muhanad); `classify.py` and prompt v1 from Session 4.
**Outputs:**
- `notebooks/04_pipeline_eval.ipynb` — fully written, with per-version score table, confusion matrices, qualitative error analysis (which frame pairs confuse the model)
- `src/elections_frames/prompts/v2.py`, `v3.py`, ... — each version with a docstring noting what it fixed relative to the previous
- One decision block in five-part form: "Frame taxonomy granularity" — informed by boundary-case patterns observed during labeling and eval (per `CLAUDE.md` workflow note: "Frame taxonomy decisions should happen *after* labeling a first batch")
- `data/processed/eval_results_v<N>.parquet` — per-version eval outputs
**Depends on:** Session 4 + external (Muhanad's labeled eval set).

### Context for resumption

`eval_set.parquet` is now hand-labeled by Muhanad and is treated as **immutable input** — never overwrite it. Read it, run the v1 prompt against it, score, then iterate. Expected iteration count: 3–5 versions. Stop iterating when marginal F1 improvement plateaus OR when the dominant error class becomes "human label is debatable" rather than "model misframed." If the latter, surface in the qualitative error analysis. The taxonomy-granularity decision block requires comparing eval scores under hypothetical merges (e.g., what if "democracy" and "process" were collapsed?) — use the eval set's confusion pattern to inform.

### Steps

1. Load `eval_set.parquet`; assert label distribution matches what was sampled in Session 3.
2. Run prompt v1 against full eval set; compute per-frame P/R/F1 and confusion matrix.
3. Qualitative error analysis: pick 20–30 disagreements, categorize the error types.
4. Iterate prompt versions, one fix at a time. Each version is its own file with a docstring noting the fix. Keep eval scores per version in a single comparison table.
5. **Frame taxonomy granularity decision block** — diagnose: what's the confusion matrix telling us? Are some frames merging in practice? Consider 5-frame vs. 6-frame vs. 7-frame alternatives via offline relabeling of the confusion matrix.
6. Final prompt version selected; document why.
7. Update `notebooks/04_pipeline_eval.ipynb` with the full narrative.

### Completion criteria

- [ ] At least 2 prompt iterations (more if eval scores demand)
- [ ] Per-version eval table committed in notebook
- [ ] Final prompt version selected and tagged
- [ ] Frame taxonomy granularity decision block complete (in `02_main.ipynb` or `04_pipeline_eval.ipynb`)
- [ ] Qualitative error analysis written (which frame pairs confuse the model and why)

### Handoff notes

*(filled during execution — note the chosen final prompt version and the precision floor that should drive Session 6's threshold pick)*

---

## Session 6: Confidence threshold + production run

**Objective:** Use the eval set to plot a precision-vs-coverage trade-off curve, pick a confidence threshold against a documented precision floor, then run the chosen prompt on the full cleaned corpus and write the production labels.
**Inputs:** Final prompt version from Session 5; cleaned articles from Session 3; eval results from Session 5.
**Outputs:**
- One decision block in five-part form: "Confidence threshold for accepting LLM labels in production"
- `data/processed/articles_classified.parquet` — production-run LLM labels (only rows above threshold)
- Updated `data/processed/llm_cost.csv` — production run cost appended
- Cost summary cell in `notebooks/02_main.ipynb`
**Depends on:** Session 5.

### Context for resumption

The chosen prompt is locked. Eval P/R/F1 are known per confidence bucket. Pick the threshold by curve, not by eye — and document the precision floor up front, before looking at the curve, to avoid post-hoc rationalization.

### Steps

1. From eval set, compute precision at varying confidence thresholds (0.5, 0.6, ..., 0.95).
2. Pick precision floor up front (suggest ≥0.85 unless argued otherwise).
3. Pick threshold = lowest confidence that meets the floor, to maximize coverage.
4. Five-part decision block: problem, diagnostic (the curve), options (3 candidate thresholds), decision + rationale, sensitivity (what does the analysis look like at ±0.05 threshold? this is also Session 8's robustness work).
5. Run final prompt on full cleaned corpus; write `articles_classified.parquet`.
6. Log cost; write running-total cell in notebook.

### Completion criteria

- [ ] Threshold decision block complete in five-part form
- [ ] `articles_classified.parquet` produced
- [ ] Cost log appended; running total visible in notebook
- [ ] Precision floor documented BEFORE the curve was examined (note this in the rationale)

### Handoff notes

*(filled during execution)*

---

## Session 7: Analysis notebook + viz module + hero figure

**Objective:** Build the actual frame-distribution analysis in `notebooks/02_main.ipynb` (by outlet origin, by election, over time around vote day), implement `viz.py` with stacked-bar and confusion-matrix helpers, produce and commit `figures/hero.png` rendered at 800×800.
**Inputs:** `articles_classified.parquet` from Session 6.
**Outputs:**
- `src/elections_frames/viz.py` — `stacked_frame_bar(df, by="outlet_origin")`, `confusion_matrix_plot(...)`, `per_election_panel(...)`, with consistent matplotlib + seaborn style
- `notebooks/02_main.ipynb` — analysis section complete (the four substantive comparisons + findings paragraphs)
- `figures/hero.png` — stacked bar, African vs. International, per election, 800×800, legible at thumbnail size
- `figures/` — other supporting figures
- Findings + limitations sections drafted in `README.md`
**Depends on:** Session 6.

### Context for resumption

All upstream pipeline data is in place. The hero figure is brand-critical: it has to read well as a LinkedIn-post thumbnail. Test the 800×800 rendering before declaring it done. The analysis is descriptive (frame distribution comparisons), not inferential — findings are stated as "outlet X allocates Y% to frame Z" with appropriate uncertainty notes, not as causal claims.

### Steps

1. Build `viz.py` helpers driven by the actual analysis needs (don't pre-build).
2. Frame distribution by outlet origin, per election — the headline comparison.
3. Frame distribution over time around vote day — 7-day rolling window or similar.
4. Frame distribution by outlet within "African" block (since CLAUDE.md flags that "African" is not a monolith).
5. Hero figure — iterate the stacked bar until it works at 800×800; commit as `figures/hero.png`.
6. Findings section: 3–5 falsifiable statements anchored in specific numbers.
7. Limitations section: GDELT coverage bias, English-only filter excluding francophone-only Senegal local coverage, LLM-as-classifier as evaluated approximation, taxonomy choice, "African" treated as a block.

### Completion criteria

- [ ] `viz.py` with at least the three helpers
- [ ] Analysis section in `02_main.ipynb` complete with all four comparisons
- [ ] `figures/hero.png` renders legibly at 800×800
- [ ] Findings + limitations sections in `README.md` drafted (not yet polished — that's Session 9)

### Handoff notes

*(filled during execution)*

---

## Session 8: Robustness notebook

**Objective:** Write `notebooks/03_robustness.ipynb` with sensitivity checks for the three decisions most likely to swing results: taxonomy granularity, confidence threshold, dedup threshold.
**Inputs:** All upstream artifacts.
**Outputs:**
- `notebooks/03_robustness.ipynb` — three sensitivity sections, each producing a small plot or table showing how the headline finding moves under perturbation
- Sensitivity rows filled in for the three relevant decisions in `README.md` decisions table
**Depends on:** Session 7.

### Context for resumption

The point of this notebook is to know — before a reader asks — which decisions the conclusions are robust to and which they aren't. Be honest: if the headline finding flips when the threshold moves by 0.05, say so prominently. That's a more credible portfolio piece than burying it.

### Steps

1. Taxonomy granularity sensitivity: re-aggregate the production data under 5-frame and 7-frame collapses; does the headline still hold?
2. Confidence threshold sensitivity: rerun the headline analysis at threshold ± 0.05; does the rank order of frames change?
3. Dedup threshold sensitivity: rerun the headline analysis at MinHash Jaccard ± 0.1; does article count change materially?
4. Write a one-paragraph summary at the top of the notebook: which decisions are headline-load-bearing.

### Completion criteria

- [ ] Three sensitivity sections in `03_robustness.ipynb`
- [ ] Decisions table in README has sensitivity column populated
- [ ] Robustness summary paragraph written

### Handoff notes

*(filled during execution)*

---

## Session 9: README polish + decisions table + final verification

**Objective:** Polish `README.md`, finalize the decisions table with all seven decisions, fill in findings and limitations, run a final consistency cross-check (every decision in the table has a corresponding five-part block somewhere in the notebooks; all done-definition items in `CLAUDE.md` are checked off), verify the hero figure renders, run smoke tests, and confirm the project would run from scratch on a fresh clone.
**Inputs:** All prior artifacts.
**Outputs:**
- `README.md` — fully polished, decisions table populated, findings + limitations final
- `CLAUDE.md` done-definition checkboxes ticked
- All smoke tests pass
- `figures/hero.png` confirmed legible
**Depends on:** Session 8.

### Context for resumption

This is the verification pass. Don't do new analysis here — only check that the existing artifacts hang together. The portfolio reader (a recruiter, a peer DS) will skim the README, look at the hero figure, glance at the decisions table, and maybe click into one notebook. The success criterion for this session is that all four of those touchpoints are coherent and self-consistent.

### Steps

1. Walk the `CLAUDE.md` Done definition checklist; tick each item or flag what's missing.
2. Populate every row of the README decisions table (chose / why / sensitivity).
3. Write findings as 3–5 falsifiable statements with numbers.
4. Write limitations frankly.
5. Cross-reference: for every decision in the table, confirm a five-part block exists in some notebook.
6. Run `pytest tests/` — all smoke tests green.
7. Render `figures/hero.png` and eyeball at 800×800.
8. Optional: pip install in a fresh venv from `pyproject.toml` and run one notebook end-to-end.

### Completion criteria

- [ ] Every `CLAUDE.md` Done-definition item ticked
- [ ] README decisions table fully populated
- [ ] Findings + limitations final
- [ ] All seven expected decision blocks present in notebooks in five-part form
- [ ] Smoke tests pass
- [ ] Hero figure verified at thumbnail size

### Handoff notes

*(final — none)*

---

## Decision & Change Log

Track decisions made during execution that affect later sessions. Each entry should note which session made the decision, what was decided, and which future sessions are affected.

| # | Session | Decision | Affects |
|---|---------|----------|---------|
| 0 | (plan) | Classifier API locked to NVIDIA NIM per README; CLAUDE.md updated 2026-05-20 to match. | 1, 4 |

## Progress Tracker

| Session | Title | Status | Date | Notes |
|---------|-------|--------|------|-------|
| 1 | Project scaffolding & setup | Not started | | |
| 2 | GDELT ingestion module | Not started | | |
| 3 | Cleaning + eval-set sampling + labeling handoff | Not started | | Triggers external dependency: Muhanad labels |
| 4 | Classifier module + prompt v1 (parallel) | Not started | | Can run in parallel with labeling |
| 5 | Eval loop + prompt iteration | Not started | | Blocked on labeled eval set |
| 6 | Confidence threshold + production run | Not started | | |
| 7 | Analysis notebook + viz module + hero figure | Not started | | |
| 8 | Robustness notebook | Not started | | |
| 9 | README polish + decisions table + final verification | Not started | | |
