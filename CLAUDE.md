# CLAUDE.md — Project 5 (Flagship): African elections frame analysis

Orientation for Claude Code working inside this project's folder. This file is self-contained because this project will eventually become its own GitHub repository.

This is the **portfolio flagship**. The pipeline + eval + analysis modularity matters more than analysis depth. Get the engineering right; analytical conclusions follow.

---

## Portfolio principles (verbatim — read these first)

### 1. Balance: Practicality ↔ Perfectionism

- Nothing is sacred as long as we are not writing code. Even the most fundamental principles can be modified if the holistic assessment results in better efficiency, reliability, and productivity.
- Every solution must be "good enough to ship and good enough to maintain" — not perfect, not rushed.
- Apply the 80/20 rule: spend effort where it yields the most real-world value.
- When perfectionism adds cost without proportional reliability gain, choose the practical path.
- Always ask: "Is this improvement worth the time, complexity, and maintenance cost?"

### 2. AI Developer Mindset (Not Pure Mathematician)

- We are engineers solving real problems — success is measured by cost, efficiency, and reliability.
- Indicators of success:
  - **Cost**: API calls, compute, tokens, dev time
  - **Efficiency**: Time-to-value, lines of code, CI/CD speed
  - **Reliability**: Error rates, test coverage, uptime, graceful degradation
- Avoid over-engineering: a working solution beats a theoretically optimal one.
- When in doubt, prototype and measure rather than analyze indefinitely.

### 3. Don't Reinvent the Wheel

- Before writing any new logic, check if it already exists in:
  - This project
  - Installed dependencies
  - MCP servers and tools available in the environment
  - Well-maintained open-source libraries
  - Claude Code slash commands, workflows, and skills
- Prefer composition over creation; integrate before building.
- Document external sources used so future contributors can maintain them.

### 4. Show Your Work (Decision Discipline) — *portfolio-specific*

Every meaningful data-processing decision — cleaning, imputing, transforming, categorizing, feature engineering, outlier handling, threshold setting, deduplication, aggregation — is presented in **five parts** in the notebook:

1. **Problem / choice point** — what needs to be decided and why it matters downstream.
2. **Diagnostic analysis** — actual code that explores the data to inform the choice.
3. **Options considered** — at least 2–3 reasonable alternatives, named explicitly.
4. **Decision + rationale** — anchored in the diagnostic, not in convention.
5. **Sensitivity** — robustness check where applicable, or explicit note that the result isn't sensitive to this choice.

No conventional choices ("just impute with mean"); every choice is **educated**. This is the lab-work signal that distinguishes a portfolio piece from a tutorial.

See `_template/notebooks/NOTEBOOK_STRUCTURE.md` for the canonical five-part block template, and `_template/src/project_name/diagnostics.py` for the helpers that operationalize the diagnostic step (`missingness_summary`, `missingness_pattern`, `distribution_summary`, `distribution_compare`, `before_after`, `compare_alternatives`).

(Within this project, the corresponding files live at `notebooks/NOTEBOOK_STRUCTURE.md` and `src/elections_frames/diagnostics.py` — the `_template/` reference above is the canonical source from the monorepo.)

This principle is portfolio-specific because the portfolio's brand is **"balance practicality vs. perfectionism — neat lab work, not ship-ready apps."** Methodological visibility is part of the deliverable.

### How these interact

Principles 1–3 are about *engineering pragmatism* — ship working things, don't over-engineer, reuse what exists. Principle 4 is about *methodological rigor* — show how data decisions were made.

There is a creative tension between #2 (AI Developer Mindset, "avoid over-engineering") and #4 (Show Your Work, "document every choice"). Resolve it this way: **be lean in the code, rigorous in the notebook**. The code can be short and pragmatic; the notebook must show the reasoning behind any non-trivial data manipulation. This is why diagnostics live in `notebooks/02_main.ipynb` (the deliverable), not in the production-style `src/` modules.

---

## Project context

**The question.** Do international media outlets frame African elections differently from African outlets — and if so, along which dimensions (security, economy, democracy, identity, process, corruption)?

**Why it matters in the LMIC / Global South frame.** Coverage framing shapes which dimensions of African political life enter the global conversation, which in turn shapes diplomatic posture, investment flows, and the conditions under which African policy debates are received outside the continent. Quantifying systematic frame divergence — if it exists — is a methodological contribution to a debate that has lived mostly in essay form. The project does **not** evaluate accuracy or quality of coverage; framing analysis is descriptive. It is also not sentiment analysis — "which dimension is foregrounded" is a different (and more tractable) question than "is the tone positive or negative."

## Data sources

| Source | Granularity | Time | Notes / caveats |
|--------|-------------|------|-----------------|
| [GDELT 2.0 GKG](http://data.gdeltproject.org/gdeltv2/) | Article-level | ±30 days around each of 4 elections | Coverage is what GDELT indexes — there is systematic bias toward what is publicly crawlable. Document this in limitations. |
| Election dates & outlet provenance | Curated | Hand-curated for this project | Lives in `data/external/outlets.csv`. Edge cases (BBC Africa service, Al Jazeera English Africa desk) are explicit columns. |
| Hand-labeled evaluation set | Article-level | 200–300 articles, stratified by election × outlet-origin × week | **MUST be hand-labeled by Muhanad.** If Claude labels the eval set, the eval is circular — Claude is grading Claude's own outputs. See "Workflow notes" below. |

**Elections covered:** Nigeria 2023, Kenya 2022, Senegal 2024, South Africa 2024.

## Planned method (one paragraph)

A four-stage pipeline: (1) **ingestion** of GDELT GKG records for ±30 days around each election; (2) **cleaning** — English-relevance filter, deduplication, outlet-origin attribution; (3) **classification** via the NVIDIA NIM API (`deepseek-ai/deepseek-v4-pro` primary, `minimaxai/minimax-m2.7` hard-failure fallback; OpenAI-compatible client at `https://integrate.api.nvidia.com/v1`; key in `../secrets.toml` under `[NVIDIA] API_KEY`) with a structured-output prompt over a 6-frame taxonomy (security / economy / democracy / identity / process / corruption); (4) **evaluation** against a hand-labeled set of 200–300 stratified articles, producing per-frame precision/recall/F1 and a confusion matrix. The production run is only treated as analytically usable once eval scores clear a documented bar. Cost is logged per run. Analysis compares frame distributions by outlet origin, by election, and over time around vote day.

## Visual style

**matplotlib + seaborn** for the stacked frame-distribution bars, the confusion matrices, and the per-election panels. Justification: the deliverables here are figures that need to render well in static thumbnail previews (LinkedIn, Twitter, recruiter-facing notebook scrubs) — stacked bars and confusion matrices are chart types that live or die on legible static rendering, not interactivity.

## Done definition

- [ ] **Ingestion pipeline**: GDELT GKG records pulled for ±30 days around each election
- [ ] **Cleaning pipeline**: filter to election-relevant English articles, deduplication, outlet attribution (African vs. International)
- [ ] **Structured analytic dataset**: parquet with `article_id, date, outlet, outlet_origin, election, themes, text_snippet`
- [ ] **Frame taxonomy**: security / economy / democracy / identity / process / corruption (codebook documented)
- [ ] **Hand-labeled eval set**: 200–300 articles, stratified by election and outlet origin — **labeled by Muhanad, not by Claude**
- [ ] **LLM-as-classifier**: prompt + structured output (NVIDIA NIM API — `deepseek-v4-pro` primary, `minimax-m2.7` fallback), with cost logged
- [ ] **Eval report**: precision/recall/F1 per frame, confusion matrix, qualitative error analysis
- [ ] **Analysis notebook**: frame distribution by outlet origin, by election, over time around vote day
- [ ] **Hero figure**: stacked bar of frame mix, African vs. international press, per election
- [ ] README with **Methodological decisions table**
- [ ] **Every data-processing decision documented in five-part form**

## Expected major decisions to document

Each one gets a five-part block inline in `notebooks/02_main.ipynb` (or `notebooks/04_pipeline_eval.ipynb` for the LLM-specific ones), and a row in the README decisions table.

1. **Outlet origin attribution** — which outlets count as African? Edge cases: BBC Africa service, Al Jazeera English Africa desk, Reuters Africa correspondents, RFI Afrique. Document the rule and the edge-case verdicts explicitly.
2. **Article-relevance filter** — which GDELT records actually concern the election? Threshold on theme tags, keyword filter, or hybrid; what does each yield in precision/recall on a small probe set.
3. **Deduplication strategy** — URL canonicalization plus near-duplicate text detection; pick a similarity threshold (e.g. MinHash Jaccard) and justify against the false-merge rate observed on a probe set.
4. **Frame taxonomy granularity** — 5 vs. 6 vs. 7 frames; what does the eval set's boundary-case distribution say about which taxonomy gives crisper labels.
5. **LLM prompt iteration** — how many revisions, what each iteration fixed (per-iteration eval scores committed). The narrative of *why each revision happened* is itself a deliverable.
6. **Confidence threshold for accepting LLM labels in the production run** — precision-vs-coverage trade-off curve on the eval set; pick the threshold that meets a documented precision floor.
7. **Eval set sampling strategy** — stratification choices (election × outlet origin × week, or some subset). Justify against the expected class-imbalance diagnostic.

## Files orientation (where to find what once scaffolded)

- `README.md` — project brief, decisions summary table
- `CLAUDE.md` — this file (orientation for future Claude Code sessions)
- `pyproject.toml` — declared dependencies; install with `pip install -e ".[viz,nlp,llm]"`
- `notebooks/02_main.ipynb` — the analysis (THE primary deliverable, decision blocks inline)
- `notebooks/03_robustness.ipynb` — sensitivity checks for taxonomy granularity, confidence threshold, dedup threshold
- `notebooks/04_pipeline_eval.ipynb` — precision/recall/F1 per frame, confusion matrix, prompt-iteration history with per-iteration scores, qualitative error analysis
- `notebooks/_scratch/01_explore.ipynb` — ugly EDA (gitignored)
- `notebooks/NOTEBOOK_STRUCTURE.md` — copy of the decision-discipline pattern (carry over from `_template/`)
- `src/elections_frames/data.py` — GDELT ingestion + caching + outlet provenance join
- `src/elections_frames/cleaning.py` — relevance filter + deduplication
- `src/elections_frames/classify.py` — NVIDIA NIM API wrapper (OpenAI-compatible client; `deepseek-v4-pro` primary, `minimax-m2.7` hard-failure fallback) with structured output + cost logging
- `src/elections_frames/viz.py` — stacked-bar and confusion-matrix helpers (matplotlib + seaborn)
- `src/elections_frames/diagnostics.py` — diagnostic helpers used inside notebook decision blocks
- `data/external/outlets.csv` — hand-curated outlet provenance (African / International + edge cases)
- `data/external/codebook.md` — frame taxonomy codebook
- `data/external/eval_set.parquet` — **hand-labeled by Muhanad** (gold set; treat as immutable input; never overwrite from pipeline output)
- `data/raw/` — GDELT raw archives (gitignored, cached locally)
- `data/processed/articles_classified.parquet` — production-run LLM labels
- `data/processed/llm_cost.csv` — running cost log per pipeline run
- `figures/` — saved figures, including `hero.png` (committed)
- `tests/test_smoke.py` — minimal smoke tests for ingestion, cleaning, classify wrapper

## Project-specific workflow notes

### ⚠️ The eval set MUST be hand-labeled by Muhanad, not by Claude

This is non-negotiable. The whole point of the eval set is to provide an *independent* ground truth against which to measure the LLM classifier's outputs. If Claude (any Claude model) generates the eval labels, then we are grading Claude's outputs against Claude's outputs — the eval is circular and the precision/recall numbers are meaningless.

Concretely, this means:
- Claude can assist by **sampling and stratifying** the 200–300 candidate articles for Muhanad to label (this is a separate task from labeling them).
- Claude can build the **labeling UI / form** if a tool is needed (a simple CSV + spreadsheet or a small Streamlit pane is fine).
- Claude can compute **inter-rater statistics** if Muhanad labels a subset twice for self-consistency.
- Claude must **never** populate the `frame_labels` column of `data/external/eval_set.parquet` programmatically with model output. That column is hand-entered by Muhanad.
- If a session is tempted to "just bootstrap the eval set with LLM labels and Muhanad will fix them later" — do not. Anchoring bias is real; pre-populated labels skew human review. Muhanad labels blind.

If you find yourself about to write code that calls the NVIDIA NIM API (or any LLM) and writes to `eval_set.parquet`, stop and re-read this section.

### Other workflow notes

- GDELT raw archives are large; pull only the ±30-day windows needed and cache aggressively. Commit a manifest of pulled file IDs to the repo.
- The NVIDIA NIM API classification step has a real dollar cost — log every run's input/output tokens to `data/processed/llm_cost.csv` and surface a running cost total in the notebook. Cost is a first-class success metric per Principle 2.
- Prompt iteration: keep every prompt version in `src/elections_frames/prompts/` with a version number. The eval notebook reads them all and produces a per-version score table. The narrative of "v3 fixed the security/identity confusion; v4 fixed the security/process confusion" is itself the methodological contribution.
- Frame taxonomy decisions should happen *after* labeling a first batch of ~50 articles. Boundary cases observed during labeling are the diagnostic that informs whether 5, 6, or 7 frames is right.
- Hero figure constraint: stacked bar of frame mix (African vs. International, per election) must read well at LinkedIn-post thumbnail size (800×800). Test the rendering before declaring it done.
- Confidence threshold for production run: do not pick by eye. Pick by precision-vs-coverage curve on the eval set, with the precision floor documented up front.
