# African Elections Frame Analysis — Multi-Session Implementation Plan

> **Created:** 2026-05-20
> **Source plan:** Synthesized from `CLAUDE.md` (Done definition + Expected major decisions) and `README.md`. No standalone plan file existed prior to this.
> **Total sessions:** 9
> **Estimated total effort:** ~9 sessions, with a hard external dependency at Session 3→5 (Muhanad hand-labels the eval set between those sessions; Session 4 can be done in parallel with that labeling).

## Overview

End-to-end build of a four-stage pipeline (ingest → clean → classify → evaluate) for measuring frame divergence between African and international press coverage of four African elections (Nigeria 2023, Kenya 2022, Senegal 2024, South Africa 2024). The flagship deliverable is the **pipeline + eval + analysis modularity** with every data-processing decision documented in five-part form inline in the notebooks.

## API choice (resolved)

The classifier uses **OpenRouter** (`deepseek/deepseek-v4-flash` primary, `minimax/minimax-m2.7` hard-failure fallback) via the OpenAI-compatible client at `https://openrouter.ai/api/v1`. API key lives in `../secrets.toml` under `[OPENROUTER] OPENROUTER_API_KEY`.

> **Migration (2026-05-22, pre-Session-5/6).** Originally the classifier targeted **NVIDIA NIM** (`deepseek-ai/deepseek-v4-pro` / `minimaxai/minimax-m2.7`). Session 4's smoke exposed free-tier latency of 23–855 s/call (mean ~328 s) and a ~37% transient-error rate — a single-threaded 250-row eval pass would run ~20 h, making prompt iteration (3–5 versions × 250 calls) the project's wall-clock bottleneck. Switched the active provider to OpenRouter Flash (verified round-trip ~4.5 s/call); a single verification call parsed cleanly. NVIDIA is retained as a **switchable but inactive** provider in `classify.py` (`provider="nvidia"`) for an easy revert. This resolves the "different endpoint" option flagged in Session 4's reliability notes below.

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

**Executed 2026-05-21. Scaffold is complete and verified.**

What's on disk:
- `pyproject.toml` — package name `elections-frames`, wheel target `src/elections_frames/`, extras `viz` (matplotlib + seaborn), `nlp` (langdetect, datasketch, scikit-learn), `llm` (openai), `dev` (ruff, pytest). Python `>=3.11`. No `tomli` dep — using stdlib `tomllib`.
- `src/elections_frames/` — `__init__.py`, `data.py` (paths constants + auto-`mkdir` only; puller goes here in Session 2), `cleaning.py`, `classify.py`, `viz.py` (with a working `save_figure`), `diagnostics.py` (all 6 helpers from `_template`: `missingness_summary`, `missingness_pattern`, `distribution_summary`, `distribution_compare`, `before_after`, `compare_alternatives`), `prompts/__init__.py`.
- `notebooks/` — `02_main.ipynb` (17 cells), `03_robustness.ipynb` (4 cells), `04_pipeline_eval.ipynb` (7 cells). All section headers per CLAUDE.md / plan. Cell IDs persisted. `NOTEBOOK_STRUCTURE.md` adapted from template with project-specific decision list. `_scratch/.gitkeep`.
- `data/external/codebook.md` — v0 with 6 frame definitions + boundary-case notes; explicit reminder of blind labeling. `outlets.csv` skeleton with the 4 named edge cases pre-populated (verdicts blank — Session 3 fills them).
- `data/{raw,processed}/.gitkeep`, `figures/.gitkeep`.
- `tests/test_smoke.py` — 5 import-only tests, all green.
- `.gitignore` — covers `data/raw/`, `data/processed/*.parquet`, `_scratch/`, caches, secrets.

Decisions made during execution (not big enough for the change log, but worth knowing):
- **English-language detector pinned to `langdetect`** (not fasttext). Rationale: GDELT records carry enough headline+lead text that langdetect's accuracy is sufficient; fasttext adds a ~125 MB model download with no benefit at this scale. If Session 3's probe set shows accuracy issues on short snippets, swap to `fasttext-langdetect` (Meta lid.176) — documented inline in `pyproject.toml`.
- **No `tomli` dep**; we read `../secrets.toml` with stdlib `tomllib` (3.11+). Saves one dep.
- **Notebooks committed as `.ipynb` only** (not jupytext-paired). User confirmed; matches "outputs are part of the GitHub deliverable" from `NOTEBOOK_STRUCTURE.md`.
- **`viz.py` already has a working `save_figure`** (matplotlib-only — no Plotly/plotnine fallback since this project committed to matplotlib + seaborn in `README.md`). The three plotting helpers (`stacked_frame_bar`, `confusion_matrix_plot`, `per_election_panel`) are intentionally not pre-built; Session 7 builds them driven by actual analysis needs.

Environment: project runs in conda env `portfolio` (Python 3.14.4) at `C:\Users\mou97\.conda\envs\portfolio\python.exe`. Editable install completed; `pytest` green (5/5). Future sessions should use the same env — do not create a `.venv`.

For Session 2 (GDELT ingestion):
- Read this handoff + the Session 2 "Context for resumption" block before starting.
- `data.py` is currently just paths + auto-`mkdir`; add the puller, manifest, and provenance join there.
- Verify election dates against public records (don't trust the rough "Nigeria 2023 / Kenya 2022 / Senegal 2024 / South Africa 2024" framing alone).
- Use the four edge-case rows in `outlets.csv` as a hint — those domains will appear in the GDELT data and need to be matchable.

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

- [x] Smoke test passes (live `GDELT_LIVE=1` round-trip green; 7/7 offline tests green)
- [x] `manifest.json` committed with file IDs (currently 4 smoke slots; will grow with background pulls)
- [x] Notebook cell shows row counts per election (added at `notebooks/02_main.ipynb` §3; reads from cache, no pull)
- [ ] All four ±30-day windows pulled and cached — **deferred to user-triggered background script `scripts/pull_all.py`** (multi-hour, multi-GB; user chose at session start to ship the module + smoke and defer the I/O wait)

### Handoff notes

**Executed 2026-05-21. Module + smoke complete; full pulls deferred to background.**

What's on disk:

- `src/elections_frames/data.py` — full implementation (~270 lines). Public API: `pull_gkg_window`, `download_slot`, `load_cached`, `outlet_provenance_join`, plus `ELECTIONS` dict and `Election` dataclass. Constants: `GDELT_BASE_URL`, `SLOT_INTERVAL_MINUTES=15`, `GKG_COLUMNS` (full 27-col schema), `KEEP_COLUMNS` (9 cols kept on load — cuts memory >50%).
- `tests/test_smoke.py` — added `test_elections_metadata`, `test_iter_slots_at_15min_boundary`, and the env-var-gated live `test_pull_gkg_window_smoke_kenya_2022`. 8 tests total: 7 pass offline, the live one passes when `GDELT_LIVE=1`.
- `scripts/pull_all.py` — CLI wrapper for the background full pull. Defaults to all four elections, ±30 days. Idempotent.
- `notebooks/02_main.ipynb` — code cell inserted after §3 ("Ingestion status") that loads cached data per election and reports rows/unique-outlets/unknown-origin/cached-yes-or-no.
- `data/raw/kenya_2022/` — 4 GKG zips (~22 MB) from the smoke pull (vote day 00:00–00:45 UTC).
- `data/raw/manifest.json` — committed; format: `{election_key: [sorted_slugs]}`. `.gitignore` updated to commit the manifest (`!data/raw/manifest.json`) while still ignoring the bulky zip archives.

Election dates (verified 2026-05-21 via WebSearch, sources cited in `data.py` docstrings):

- Nigeria 2023 presidential: **2023-02-25** (Tinubu / APC)
- Kenya 2022 general: **2022-08-09** (Ruto / UDA)
- Senegal 2024 presidential: **2024-03-24** (Faye / PASTEF; postponed from original 2024-02-25 — comment in code explains the choice to center the window on actual vote day, not original schedule)
- South Africa 2024 general: **2024-05-29** (ANC lost majority for first time since 1994)

Smoke-pull data quality sanity check (Kenya 2022, 1 hour of vote day, 4 slots):

- 5,519 records, 1,156 unique outlets — GKG firehose is fat (~5,500 rows/hour even on a single hour)
- Top sources are predictable aggregators (iHeart, Yahoo, MSN, MENAFN, Daily Mail, etc.)
- Provenance join already matched 10 rows against the four `outlets.csv` edge cases — Session 3's outlet attribution decision block has actual rows to chew on, which is reassuring

Decisions made during execution (worth knowing, not big enough for the change log):

- **`outlet_provenance_join` is a Session-2 pragmatic stub.** It does a simple bare-domain substring match against `outlets.csv`. The four named edge cases have full paths in the sheet (`bbc.com/news/world/africa`, etc.); the current match only uses the bare domain (`bbc.com`), so BBC main vs. BBC Africa is not yet distinguished. **Session 3 must refine this** — that's exactly what the outlet-attribution decision block is for. Path-sensitive matching is needed for the four edge cases plus bulk resolution of the long tail of unknowns.
- **Column reduction at load time.** `load_cached` defaults to keeping only 9 of the 27 GKG columns (GCAM, embed columns, etc. dropped). Saves >50% RAM on a full corpus. Pass `keep_columns=` to override if Session 3 wants tone/locations data the default already includes the most useful fields.
- **`DATE` parsing.** UTC datetime; rows that fail to parse are dropped silently — GKG occasionally has malformed lines, and `on_bad_lines="skip"` + `errors="replace"` at the CSV level plus `errors="coerce"` at the datetime level means we drop bad rows rather than crash. Acceptable for this volume; documented inline.
- **CACHE_DIR (project root `.cache/`) is created but unused.** Carry-over from Session 1 scaffold; harmless. Future sessions may use it for the LLM response cache.

Disk-usage expectation for the background full pull: roughly **5.5 MB compressed per 15-min slot × 96 slots/day × 61 days × 4 elections ≈ 130 GB total**. Make sure there's enough free disk before kicking it off. If disk-constrained, the manifest + downstream parquets are the ground-truth artifacts — the raw zips can be deleted and re-pulled at any time.

For Session 3:

1. **Run the background pull first.** `python scripts/pull_all.py` from project root. Expect hours; idempotent so it's safe to restart. The notebook §3 cell will populate row counts as elections finish.
2. The outlet-attribution decision block is now empirically grounded — `load_cached(election).SourceCommonName.value_counts()` will show the unknown long tail that needs resolving. The four edge-case rows already matched in the smoke pull are a sanity check that the join wiring works; the rule + verdicts are still Session 3's job.
3. `pyproject.toml` already has `langdetect`, `datasketch`, `scikit-learn` in the `nlp` extra — install with `pip install -e ".[nlp]"` before starting Session 3.
4. `outlet_provenance_join` will need a refined version. Consider exposing it as `cleaning.attribute_outlet_origin` (in `cleaning.py`) rather than expanding `data.py` — `data.py` is the I/O layer, `cleaning.py` is the cleanup layer.

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

**Executed 2026-05-21. Cleaning pipeline complete; 250 stratified eval candidates ready for Muhanad to hand-label.**

What's on disk:

- `src/elections_frames/cleaning.py` (~620 lines). Public surface: `attribute_outlet_origin`, `collapse_edge_to_international`, `filter_english`, `filter_relevant`, `deduplicate`, `canonicalize_url`, `build_text_snippet`, `load_probe`, `iter_cached_zips`, `run_pipeline_election`, `sample_eval_candidates`. Streams a single election through the full pipeline in `batch_size=200` chunks (≈ 2 days of GKG slots per batch) so memory stays bounded.
- `data/external/outlets.csv` — finalized. Four named edge cases have verdicts in the file; rule + rationale documented column-by-column. Path-based detection works only for BBC Africa (`bbc.com/news/world-africa-*`) and RFI Afrique (`rfi.fr/fr/afrique`, `rfi.fr/en/africa`); AJE and Reuters URL paths don't separate region from main feed (empirical finding from probe), so those fold to International by publisher-origin rule.
- `data/processed/articles_clean.parquet` — **79,372 cleaned articles** (combined; per-election parquets also written). Schema: 9 GDELT columns + `election`, `outlet_origin`, `text_snippet`.
- `data/processed/pipeline_counts.csv` — per-election row-count progression (raw → relevant → english → dedup) + per-run timing.
- `data/external/eval_set_candidates.parquet` — **250 stratified candidates**, 124 African / 126 International, 40 strata (4 elections × 2 origins × 5 week-buckets) populated 6–8 each. All `frame_labels` blank — Muhanad fills these in via the labeling notebook.
- `notebooks/02_main.ipynb` — three full five-part decision blocks (outlet attribution, relevance filter, dedup) + structured-dataset preview + eval-sampling decision block. All cells are pre-wired against the artifacts above; the notebook reads end-to-end without re-running the pipeline (just `articles_clean.parquet` reads).
- `notebooks/eval_labeling.ipynb` — ipywidgets-based labeling UI. No LLM suggestions visible. Saves to `data/external/eval_set.parquet` after every label; resumable.
- `docs/labeling_handoff.md` — 1-page guide for Muhanad.
- `scripts/run_cleaning.py` — CLI wrapper for the multi-hour pipeline run (idempotent).
- `tests/test_smoke.py` — 5 new cleaning-module tests (outlet attribution, word-boundary keyword match, URL canonicalization, dedup syndication, sampling schema). **All 15 offline tests green; 2 live tests skipped as designed.**

**Per-election pipeline counts (full corpus, hybrid relevance + dedup@0.8):**

| Election | Raw rows | Relevant | English | After dedup | Wall time |
|---|---:|---:|---:|---:|---:|
| Nigeria 2023 | 6,621,830 | 40,776 | 40,223 | 29,724 | 68.7 min |
| Kenya 2022 | 6,587,750 | 10,147 | 9,955 | 6,119 | 58.8 min |
| Senegal 2024 | 8,875,777 | 4,400 | 4,320 | 3,820 | 73.6 min |
| South Africa 2024 | 8,604,730 | 44,043 | 43,740 | 39,709 | 87.2 min |
| **TOTAL** | **30,690,087** | **99,366** | **98,238** | **79,372** | **4.8 hrs** |

**Per-election × origin split (cleaned corpus, edge labels folded into International for headline analysis):**

| Election | African | International | Total |
|---|---:|---:|---:|
| Nigeria 2023 | 14,755 | 14,969 | 29,724 |
| Kenya 2022 | 3,778 | 2,341 | 6,119 |
| Senegal 2024 | 585 | 3,235 | 3,820 |
| South Africa 2024 | 9,147 | 30,562 | 39,709 |

Senegal's African-outlet count (585) is the smallest by far — the English-language filter is excluding the French and Wolof-language Senegalese press. **This is a real limitation** and should be called out in Section 10 of `02_main.ipynb` (Limitations). Senegal coverage is reachable via international wires (`reuters.com`, `aljazeera.com`, etc.) but local Senegalese framing is largely missing.

Decisions logged during execution:

- **(Decision Log #1) Classification text = GKG metadata only** (URL title-slug + V21AllNames + V2EnhancedThemes). No live HTML scraping. Documented in `outlets.csv` and the cleaning module. Trade reliability + reproducibility for some classifier precision.
- **(Decision Log #2) Labeling UI = ipywidgets, not Streamlit.** Same env as everything else, no port management.
- **Edge-case rule:** only BBC Africa and RFI Afrique are URL-detectable. AJE and Reuters URL paths don't separate region; they roll up to International by publisher-origin rule. `Edge_BBC_Africa` (27 rows in corpus) and `Edge_RFI_Afrique` (73 rows) labels are retained for the robustness check (re-folding as African) — see `03_robustness.ipynb` plan.
- **Keyword broadening + word-boundary regex.** Initial substring matching produced 40–70% false positives (`avocado` matched `Ba`, `bafana` matched `Ba`, etc.). Word-boundary fix collapsed false positives to ~10–15% (eyeball precision check). Keyword sets in `data.py` were broadened from 5–7 to 10–14 tokens each.
- **Pipeline ordering: relevance filter before English filter.** Vectorized regex on 1.1M rows/batch vs. row-wise `langdetect` on ~1,500 rows/batch — major speedup.
- **Dedup threshold 0.8.** Decision was not sensitive (0.7–0.95 differed by <3% on the probe; no false merges observed at any threshold).

**Cross-election contamination, known and logged:**

`"Labour Party"` (Nigeria 2023 keyword) matched UK Labour Party coverage during the same window. Dedup correctly collapses syndicated wire copy of UK politics, but the residual UK-politics rows remain in the corpus. This is partly why Nigeria 2023's `International` share is so high (14,969 vs. 14,755 African — atypical balance). The LLM classifier in Session 5 will get a third opportunity to filter these out via the "is this article actually about the named election?" check we can build into the prompt.

For Session 5 (eval loop + prompt iteration), when Muhanad finishes labeling:

1. **Read `data/external/eval_set.parquet`.** Treat as immutable — never overwrite from pipeline output.
2. **Class-imbalance diagnostic first.** Per-frame counts from the hand labels will tell us whether stratification is OK or whether the eval set is heavily skewed to one or two frames. If skewed, top up via additional candidates (the candidate file has columns ready for this; we just sample more).
3. **First prompt iteration target: "is this article about the named election?"** Add a relevance check to prompt v2 — return an `off_topic` flag when the article is e.g. UK Labour Party coverage that slipped through. The eval set's `too_thin=True` rows are exactly the kind of cross-election contamination the prompt should catch.
4. **The classifier already takes GKG metadata as input** (Session-4 smoke confirms this works), so no plumbing changes are needed for Session 5 — just iterate the prompt against `eval_set.parquet`.
5. **Use the labeling notebook's `labeler_notes` column.** It contains Muhanad's free-text observations on boundary cases (security ↔ process, democracy ↔ process, etc.) — these are direct input to the taxonomy granularity decision block in `04_pipeline_eval.ipynb`.

For Session 7 (analysis + hero figure):

- Senegal's small African-outlet count means the Senegal panel in the hero figure will be visually thin on the African side. Plan: either (a) keep the four-panel grid honest and let Senegal show what it shows, or (b) annotate Senegal's African bar with its sample size. Decide in Session 7.
- The intra-African breakdown (Section 7 #3 of `02_main.ipynb`) is now empirically grounded — 28,265 African rows give enough headroom for a per-country breakdown within African.

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

- [x] `classify_article` returns valid structured output for 5 articles (live `NVIDIA_LIVE=1` smoke green — all 5 parsed + validated)
- [x] `llm_cost.csv` initialized + populated with 9 rows (5 successful + 3 transient retries + 1 retry-then-success) across 5 articles
- [x] Mocked test passes (3 mocked tests covering success, parse-error fallthrough, and prompt-message structure); live test passes when `NVIDIA_LIVE=1`
- [x] Prompt v1 saved at `src/elections_frames/prompts/v1.py` with docstring noting "v1 — initial draft, no eval data yet"

### Handoff notes

**Executed 2026-05-21. Classifier + prompt v1 complete; verified end-to-end against the real NVIDIA NIM endpoint.**

What's on disk:

- `src/elections_frames/classify.py` (~250 lines). Public surface: `FrameClassification` (Pydantic schema), `ClassifyResult` (dataclass with parsed output + per-call metadata), `classify_article(text, ...)`. Internal: secrets loader (env var or `../secrets.toml`), OpenAI-compatible client builder, append-only cost-log writer.
- `src/elections_frames/prompts/v1.py` — codebook-grounded system prompt (~4.8K chars) + `render_messages(text)` helper. Three synthesized few-shot examples (security+process, economy, democracy). No content from `eval_set.parquet` or any specific real article.
- `src/elections_frames/prompts/__init__.py` — `get(version)` import helper so `classify.py` drives any version uniformly.
- `tests/test_classify_smoke.py` — 4 tests: mocked success path, mocked parse-error fallthrough (verifies retry + fallback wiring), prompt structure check, env-var-gated live round-trip. All 3 offline pass; live passes with `NVIDIA_LIVE=1`.
- `data/processed/llm_cost.csv` — initialized; 9 rows from the live smoke (5 articles, 3 retries needed for transient errors).

Live smoke results (Kenya 2022, 5 GKG metadata descriptors):

| # | Primary | Confidence | Frames | Tokens (in/out) | Duration |
|---|---------|-----------:|--------|-----------------|---------:|
| 1 | process | 0.10 | [process] | 2499/82 | 855s |
| 2 | identity | 0.90 | [identity, security] | 2958/94 | 32s |
| 3 | economy | 0.70 | [economy] | 1147/94 | 397s |
| 4 | economy | 0.90 | [economy, corruption] | 1175/96 | 334s |
| 5 | process | 0.10 | [process] | 1584/80 | 23s |

Decisions made during execution:

- **Text-source agnosticism.** Per user choice at Session 4 start, `classify_article` accepts `text: str` with no opinion on whether that text is article body, GKG metadata, or hybrid. The production `text_snippet` definition is **Session 3's decision** — to be a five-part block in `02_main.ipynb`. Two of the five smoke classifications returned `primary=process, conf=0.10` (the "input too thin" escape valve in the prompt) — direct evidence that GKG-metadata-only descriptors lose framing signal. This is useful input for the Session 3 decision: pure metadata is likely insufficient; URL-scraping or a hybrid is probably needed.
- **Cost units.** Logging input/output tokens + duration_seconds only — no USD column yet. Per-token rates for NVIDIA NIM hosted DeepSeek vary; analysis-time conversion (in `04_pipeline_eval.ipynb`) is cleaner than baking in rates now.
- **DeepSeek thinking mode = off.** `extra_body={"chat_template_kwargs": {"thinking": False}}` matches the project's sample script and saves token cost. Session 5 should A/B `thinking=True` against eval scores — `classify_article` exposes the param.
- **Temperature = 0.0** for deterministic classification (overriding the `temperature=1` in the sample script, which is wrong for a classifier).
- **`response_format={"type": "json_object"}`** — standard OpenAI-compatible JSON mode. NIM supports it. Pydantic validates on our side.
- **Retry policy.** 2 retries with exponential backoff on `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`. Hard errors (`BadRequestError`, generic `APIError`) skip retries and fall straight to the secondary model. Parse errors get retries too (intermittent NIM JSON-mode hiccups should be rare but recoverable).
- **Cost log path is parameterizable** — `cost_log_path=` kwarg lets tests use `tmp_path` to avoid polluting the real log.

API reliability observations (important for Session 6 planning):

- **NVIDIA NIM `deepseek-v4-pro` is unreliable + slow on the free tier.** Across 8 total attempts for 5 articles, 3 transient `InternalServerError`s (retried successfully) and per-call durations spanning 23s – 855s. Mean: ~328s. The 14-minute cold-start on call 1 + the second-batch retries dominate.
- **Implication for Session 6:** at this latency a 200-article eval pass takes ~18 hours single-threaded, and a full corpus production run is multi-day. Session 6 should plan for either parallel workers (multiprocessing on classify_article), batching if NIM supports it (the OpenAI SDK has `batch` for it), or a different endpoint. None of that needs to be solved in Session 4 — but the cost log will surface these distributions once a real corpus exists.
- **No fallback to `minimax-m2.7` was triggered in the smoke** — primary always recovered. Fallback wiring is exercised by `test_classify_mocked_invalid_frame_value` (mocked), so it's verified at the code-path level even though the live test didn't need it.

For Session 5 (after Muhanad's labels arrive):

1. **Background GDELT pull status check.** Before iterating prompts, confirm `scripts/pull_all.py` is making progress — Session 5 needs at least the eval-set sample to be drawn from real cleaned data (which is Session 3's job, also blocked on the full pull).
2. **First iteration target: address the "input too thin" failure mode.** Two of five smoke results landed in the escape valve. Once `text_snippet` is real article text (Session 3 decision), this should largely disappear — but if it doesn't, that's prompt v2's first fix.
3. **Thinking mode A/B test.** `thinking=True` vs `False` on the eval set: does it close the precision gap on boundary cases (security↔process, democracy↔process)? Probably worth one prompt iteration.
4. **Watch for the security/process and democracy/process confusions** — those are flagged in `codebook.md` as boundary cases. The two `[process] conf=0.10` results in the smoke may be the model defaulting to process when uncertain; prompt v2 might need a sharper "don't default to process" instruction.
5. **Cost-log analysis tooling.** When Session 5 generates per-version eval runs, `pd.read_csv(COST_LOG_PATH)` is enough — no need for fancier infra. Group by `run_id` + `prompt_version`.

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

- [x] At least 2 prompt iterations (4 versions: v1→v2→v3→v4)
- [x] Per-version eval table committed in notebook (`04_pipeline_eval.ipynb` §2)
- [x] Final prompt version selected and tagged (**v3** on `deepseek-v4-flash`)
- [x] Frame taxonomy granularity decision block complete (`04_pipeline_eval.ipynb`, five-part)
- [x] Qualitative error analysis written (§4 — false-abstain + democracy↔process seam)

### Handoff notes

**Executed 2026-05-22. Eval loop + 4 prompt versions + model A/B complete. Production config: prompt v3 on `deepseek-v4-flash`.**

What's on disk:

- `src/elections_frames/eval.py` (~210 lines) — scoring module (kept in `src/`, it's reusable + unit-tested). Public surface: `to_frame_set`, `per_frame_scores(gold, pred, frames=FRAMES)`, `run_summary`, `confusion_matrix_primary`, `compare_versions`, `exact_match_ratio`. Multi-label-with-abstention metric: per-frame binary P/R/F1 over the 6 frames (n=250; empty gold = all-negatives), explicit **abstain** P/R, single-label confusion (6 frames + `none`). `per_frame_scores` takes a `frames=` arg so merged taxonomies are scored honestly (raises if a label would be dropped).
- `tests/test_eval.py` — 7 tests; `tests/test_classify_smoke.py` — +2 (abstain schema). **26 offline tests green, 2 live skipped. ruff clean.**
- `src/elections_frames/classify.py` — `FrameClassification` schema **relaxed to allow abstention**: `frames` may be empty, `primary_frame` may be `None`; a `model_validator` reconciles them (lenient, non-raising) and adds an `.abstained` property. Invalid-vocab values still hard-fail. v1's behavior unchanged.
- `src/elections_frames/prompts/v2.py`, `v3.py`, `v4.py` — each a single isolated change with a docstring stating what it fixed.
- `scripts/run_eval.py` — reusable runner: `python scripts/run_eval.py <version> [--model …] [--limit N] [--thinking]`. Never writes `eval_set.parquet`. Tags outputs with a model slug for non-default models.
- `data/processed/eval_results_v{1,2,3,4}.parquet` (deepseek) + `eval_results_v{3,4}_minimax-m2-7.parquet` (model A/B). Each has gold + pred frames/primary/confidence/rationale.
- `notebooks/04_pipeline_eval.ipynb` — fully written + executed (26 cells, outputs committed): class-imbalance diagnostic, per-version table, model A/B, confusion matrices, qualitative error analysis, taxonomy-granularity five-part decision block, final selection + handoff.
- `figures/eval_confusion_v1_v3.png` — committed.

**The prompt-iteration narrative (deepseek-v4-flash, micro-F1):**

| Version | Single change | micro-F1 | macro-F1 | abstain-F1 |
|---|---|---:|---:|---:|
| v1 | initial; cannot abstain (rule: thin → `["process"]`) | 0.441 | 0.413 | 0.00 |
| v2 | **enable + instruct abstention** (empty frames OK) | 0.523 | 0.524 | 0.768 |
| **v3** ✅ | few-shot examples rewritten in **GKG-metadata format** | **0.552** | 0.519 | 0.737 |
| v4 | democracy↔process disambiguation rule | 0.522 | 0.515 | 0.740 |

v1→v3 is monotone; **v4 regressed** (the dem↔proc rule just trades errors across the seam) — kept as the probe showing the seam is taxonomy-inherent. **v3 selected.**

**Key findings for downstream sessions:**

1. **54% of eval rows (135/250) carry no frame.** Muhanad, labeling blind, judged most GKG-metadata-only snippets too thin to foreground a frame. This caps achievable F1 — the ceiling is the input (metadata-only, Decision Log #1), not the prompt. Restate in `02_main.ipynb` §Limitations + README.
2. **Model A/B: the cheaper `deepseek-v4-flash` beats `minimax-m2.7` on every axis** — accuracy (v3: 0.552 vs 0.460; v4: 0.522 vs 0.474), latency (~6.5 s vs ~13 s/call median), and reliability (≈4–5 vs ≈22–28 transient errors per 250-row pass). The bigger model over-frames thin metadata (lower precision). **Production stays on deepseek-flash.**
3. **Taxonomy: keep 6 frames for the headline.** The democracy↔process seam is the dominant residual confusion (merging it lifts micro-F1 0.552→0.598), but collapsing it would delete the procedural-vs-institutional distinction the research question is about. The 5-frame `democracy+process→governance` collapse is carried as the **Session 8 robustness alternative**.

For Session 6 (confidence threshold + production run):

1. **Production config is locked: prompt `v3`, model `deepseek/deepseek-v4-flash`.** Run on `data/processed/articles_clean.parquet` (79,372 rows) via `classify_batch`.
2. **Threshold on the curve, not by eye.** Use the v3 eval predictions' `pred_confidence` vs gold. Among **framed** v3 predictions (n=119) confidence mean ≈0.724. Recommend applying the threshold to *framed* predictions only (abstentions pass through as "no frame"); document a precision floor *before* viewing the curve.
3. **Cost-log hygiene:** re-running a version reuses its fixed `run_id` and appends, so `llm_cost.csv` can hold duplicate ok-rows (eval_v4 has some). Dedup by `(run_id, article_id)` for cost accounting, OR give the production run a unique `run_id` (`classify_batch` does this when `run_id=None`). The eval notebook already dedups for display.
4. **`thinking=True` A/B was not run** (Session 4 flagged it as optional). Deferred — v3/deepseek already clears the practical bar; revisit only if Session 6 wants more precision headroom.

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

- [x] Threshold decision block complete in five-part form (`04_pipeline_eval.ipynb` §6, executed; curve `figures/threshold_curve.png`)
- [ ] `articles_classified.parquet` produced — **runner shipped + smoke-verified; the multi-hour run is deferred to the user** (`scripts/run_production.py`)
- [x] Cost log + running-total cell added (`02_main.ipynb` §6 — shows eval spend + production projection now; auto-fills production cost once the run lands `prod_*` rows)
- [x] Precision floor documented BEFORE the curve was examined (pre-registered ≥0.85 in the plan, 2026-05-20; the block confronts it with the curve)

### Handoff notes

**Executed 2026-05-22. Threshold decision block + production runner complete; the production API run is deferred to the user (their choice — "I'll run it later myself").**

What's on disk:

- `notebooks/04_pipeline_eval.ipynb` §6 — the five-part **confidence-threshold decision block** (executed, outputs committed). Computes the precision-vs-coverage curve on the v3 eval predictions, the frame-mix-skew check, and per-frame precision ungated vs gated. `figures/threshold_curve.png` committed.
- `scripts/run_production.py` — idempotent / resumable production runner. Prompt v3 / deepseek-flash, `classify_batch` over `articles_clean.parquet`, checkpoints the output parquet after every chunk, skips already-done `GKGRECORDID`s on resume, `--elections` to stage by country, `--limit` for smoke. Writes `articles_classified.parquet` with raw labels + `pred_confidence` + an `accepted` flag. ruff clean. Validated live on a partial Nigeria-2023 run (`run_id=prod_e7cd7a1870c8`, 190 rows, all `ok`, ~$0.06 real spend) — those `prod_*` rows are **kept** in `llm_cost.csv` as genuine spend; the output parquet is gitignored so the cost log is the only committed trace. The full multi-hour run across all four elections is still deferred to the user.
- `notebooks/02_main.ipynb` §6 — fixed the section header, replaced the threshold placeholder with a decision **summary** (full block lives in 04 §6), added a graceful **production-results** cell and a **running-cost** cell. Both execute before the production run exists (results cell prints how to produce the file; cost cell shows eval spend + projected production cost).

**The threshold decision (the important call):**

- Pre-registered floor was **≥ 0.85** (plan, 2026-05-20, before eval data). The curve shows it is **unreachable at usable coverage**: precision tops out at 0.833 on **5 articles** (conf ≥ 0.90). The GKG-metadata-only input (Decision Log #1) caps precision in the high-0.60s. Per the global "common sense over the contract" directive, we relaxed the floor and documented it.
- Chosen operating point: the **0.75 confidence elbow** (precision 0.547 → 0.684; retains 53 of 119 framed eval rows).
- **Critical finding:** the gate is **not frame-neutral** — at 0.75 the primary-frame mix shifts **+15 pp toward `democracy`, −10 pp away from `process`**. Since the headline finding *is* a frame distribution, gating the headline would bias the dependent variable.
- **Resolution:** the threshold is stored as an **`accepted` flag, not a filter**. Production writes every row (framed + abstained) with its confidence and `accepted = framed AND conf ≥ 0.75`. This is strictly more flexible than the plan's "only rows above threshold" wording, and is **required** by Session 8's mandated threshold ±0.05 sensitivity rerun (you can't re-threshold what you've dropped).

For the user — to produce the production dataset:

1. `python scripts/run_production.py` from the project root. ~$23, ~9–18 h at 8–16 workers (`--max-workers 16` to halve wall-time). Resumable: safe to Ctrl-C and restart; it skips done rows. Stage with `--elections kenya_2022` (smallest, ~6 k rows, ~$2) to validate the full path cheaply before the big run, then run the rest.
2. When it finishes, re-run `02_main.ipynb` §6 — the results + cost cells auto-populate (coverage by election, frame-mix teaser, production cost from the `prod_*` rows).

**Known issue (flag for Session 9's end-to-end run):** `02_main.ipynb` is committed **unexecuted** (as it was after Session 3 — no cell outputs). Re-running it top-to-bottom currently **times out on the §3 "Ingestion status" cell** (a Session-2 cell), because that cell calls `load_cached()` over the now-full raw corpus (~30M rows across the 4 elections) just to print row counts. It was cheap when the raw cache was nearly empty; it is multi-GB / >15 min now. Options for Session 7/9: guard it behind a flag, sample/stride the load, or read counts from `pipeline_counts.csv` / `articles_clean.parquet` instead of raw. The Session-6 §6 cells themselves are validated to run standalone (results cell degrades gracefully pre-production; cost cell shows eval spend + ~$23 projection).

For Session 7 (analysis + hero figure):

1. `articles_classified.parquet` carries `election, outlet_origin, DATE, SourceCommonName` + `pred_frames, pred_primary, pred_confidence, pred_abstained, accepted, model_used`. Join back to `articles_clean.parquet` on `GKGRECORDID` if you need `text_snippet` / themes.
2. **Decide all-framed vs. accepted-only for the headline.** The frame-mix skew above means the choice matters; the `accepted` column lets you do either. Recommend: headline on **all framed** labels (preserves the distribution), with `accepted`-only as a robustness lens (Session 8).
3. Senegal's thin African side (585 rows, Session 3 handoff) still applies — the hero figure's Senegal/African bar will be small.

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
| 1 | 3 | Classification text is assembled from GKG metadata only (V2EnhancedThemes + V21AllNames + V2Tone + URL title-slug) — no live HTML scraping. Trade reliability + reproducibility for some classifier precision; document the trade-off in limitations. | 3, 5, 6 |
| 2 | 3 | Labeling UI is an ipywidgets-based form inside `notebooks/eval_labeling.ipynb` — no Streamlit. | 3 |
| 3 | (pre-5/6, 2026-05-22) | **Classifier provider switched NVIDIA NIM → OpenRouter** (`deepseek/deepseek-v4-flash` primary, `minimax/minimax-m2.7` fallback, base_url `https://openrouter.ai/api/v1`, key `[OPENROUTER] OPENROUTER_API_KEY`). Driver: Session 4's free-tier NIM latency (23–855 s/call, ~37% transient errors → ~20 h single-threaded eval pass) made prompt iteration the project bottleneck. Verified OpenRouter round-trip ~4.5 s/call. NIM kept switchable-but-inactive (`provider="nvidia"`). Live test gate renamed `NVIDIA_LIVE` → `OPENROUTER_LIVE`. Also added `classify_batch` — a thread-pooled fan-out over `classify_article` (input-order results, per-item failures captured not fatal, lock-serialized cost-log appends; live-verified ~fully parallel across workers). Together these resolve **both** the "different endpoint" and "parallel workers" options flagged in Session 4. | 5, 6 |
| 4 | 5 | **Schema relaxed to allow abstention.** `FrameClassification.frames` may be empty (`primary_frame=None`) to mean "no frame clearly foregrounded", matching how 135/250 eval rows were hand-labeled. Loosened from Session-4 `min_length=1`; invalid-vocab still hard-fails; v1 behavior unchanged. | 5, 6 |
| 5 | 5 | **Production classifier config = prompt `v3` + model `deepseek/deepseek-v4-flash`.** v3 (metadata-format few-shots) is best of 4 versions (micro-F1 0.552). Model A/B: deepseek beats minimax-m2.7 on accuracy, latency, and reliability — minimax stays fallback-only. | 6, 7 |
| 6 | 5 | **Taxonomy: 6 frames for the headline analysis; `democracy+process→governance` 5-frame collapse carried as the Session-8 robustness alternative.** Democracy↔process is the dominant seam (merge lifts micro-F1 0.552→0.598) but is the substantive distinction the research question targets, so it is stress-tested, not assumed away. | 7, 8 |
| 7 | 6 | **Confidence threshold = 0.75, stored as an `accepted` flag, NOT a filter.** Pre-registered ≥0.85 floor is unreachable (metadata-only input caps precision in the high-0.60s; ≥0.85 survives on 5/250 eval rows), so it was relaxed to the 0.75 precision elbow and documented. Crucially, gating at 0.75 is **not frame-neutral** (+15pp democracy / −10pp process), so `articles_classified.parquet` keeps **all** rows + `pred_confidence` + `accepted = framed AND conf≥0.75`. Headline all-framed-vs-accepted is a Session-7 choice; Session-8's ±0.05 sensitivity requires the un-dropped rows. Supersedes the Session-6 plan wording "only rows above threshold". | 7, 8 |

## Progress Tracker

| Session | Title | Status | Date | Notes |
|---------|-------|--------|------|-------|
| 1 | Project scaffolding & setup | Complete | 2026-05-21 | 5/5 smoke tests green; env: conda `portfolio` (3.14.4) |
| 2 | GDELT ingestion module | Complete (module) / pulls deferred | 2026-05-21 | Module + smoke + manifest green. Full pulls deferred to `scripts/pull_all.py` (user-run, background). |
| 3 | Cleaning + eval-set sampling + labeling handoff | Complete | 2026-05-21 | 79,372 cleaned articles · 250 stratified eval candidates · 3 five-part decision blocks + labeling UI shipped. Pipeline run: 4.8h, idempotent. |
| 4 | Classifier module + prompt v1 (parallel) | Complete | 2026-05-21 | Live smoke green on 5 articles. NIM latency 23-855s/call, 37% transient error rate; retry+fallback wiring exercised. |
| 5 | Eval loop + prompt iteration | Complete | 2026-05-22 | 4 prompt versions (v1→v4); v3 selected (micro-F1 0.552). Model A/B: deepseek > minimax. Taxonomy: 6 frames. eval.py + tests; notebook executed. 26 tests green. |
| 6 | Confidence threshold + production run | Complete (deliverables) / prod run deferred | 2026-05-22 | Threshold block (04 §6) + curve figure + `run_production.py` (smoke-verified) + cost cell (02 §6). Threshold 0.75 stored as `accepted` flag, not filter (gate not frame-neutral). ~$23/~9-18h API run deferred to user. |
| 7 | Analysis notebook + viz module + hero figure | Not started | | |
| 8 | Robustness notebook | Not started | | |
| 9 | README polish + decisions table + final verification | Not started | | |
