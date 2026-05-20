# African elections frame analysis

> Do international and African outlets cover the same election as if they were two different events?

![Hero figure](figures/hero.png)

## The question

Four recent African elections — Nigeria 2023, Kenya 2022, Senegal 2024, South Africa 2024 — were covered simultaneously by African and international press. This project asks whether the *frames* applied to those elections differ systematically by outlet origin, and along which dimensions: security, economy, democracy, identity, process, or corruption. We are **not** evaluating coverage quality or accuracy; framing analysis is descriptive, not normative. We are also **not** running a sentiment analysis — frame is about *which dimension of the story is foregrounded*, which is a different and more analytically tractable question than "is the coverage positive or negative."

The flagship deliverable here is not the headline number — it's the **pipeline + eval + analysis modularity**: an LLM-classification system whose outputs have been validated against a hand-labeled gold set.

## Data

| Source | Granularity | Time coverage | Access |
|--------|-------------|---------------|--------|
| [GDELT 2.0 Global Knowledge Graph](http://data.gdeltproject.org/gdeltv2/) | Article-level | ±30 days around each election | Public (raw archives) |
| Election dates & outlet provenance | Per outlet | Hand-curated for this project | Curated in `data/external/outlets.csv` |
| Hand-labeled evaluation set | Article-level | 200–300 articles, stratified | **Labeled by Muhanad** — gold set, not LLM-generated |

Access date: planned 2026-05-XX (to be filled when ingestion notebook is first run).

## Method

A four-stage pipeline. **(1) Ingestion** pulls GDELT GKG records for the ±30-day window around each of the four elections. **(2) Cleaning** filters to election-relevant English-language articles, deduplicates, and attributes each outlet to "African" or "International" using a hand-curated provenance list. **(3) Classification** uses the NVIDIA NIM API (`deepseek-ai/deepseek-v4-pro` primary, `minimaxai/minimax-m2.7` hard-failure fallback) with a structured-output prompt to assign one or more frames per article from a 6-frame taxonomy (security / economy / democracy / identity / process / corruption). **(4) Evaluation** compares LLM labels against a hand-labeled set of 200–300 stratified articles, computing precision/recall/F1 per frame and a confusion matrix. Only after the eval passes a documented quality bar do we treat the production-run labels as analytically usable. The strongest critique of this design is that the frame taxonomy is itself a choice — different taxonomies would yield different distributions. The taxonomy is documented in a codebook (`data/external/codebook.md`) and its granularity is one of the documented decisions below.

## Methodological decisions

Each major data-processing decision was made by **diagnostic first, choice second**. The table below is an at-a-glance summary; the full five-part rationale (problem / diagnostic / options / decision + rationale / sensitivity) lives inline in `notebooks/02_main.ipynb`.

| Decision | Chose | Why (anchored in diagnostic) | Sensitivity |
|----------|-------|------------------------------|-------------|
| Outlet origin attribution (African vs. International; edge cases like BBC Africa service, Al Jazeera English Africa desk) | *to be filled during implementation* | *anchored in diagnostic — see notebook §3* | *to be filled* |
| Article-relevance filter (theme-tag threshold vs. keyword filter vs. hybrid) | *to be filled during implementation* | *anchored in diagnostic — see notebook §3* | *to be filled* |
| Deduplication strategy (URL canonicalization + near-duplicate text-similarity threshold) | *to be filled during implementation* | *anchored in diagnostic — see notebook §4* | *to be filled* |
| Frame taxonomy granularity (5 vs. 6 vs. 7 frames; boundary cases) | *to be filled during implementation* | *anchored in eval-set boundary diagnostics — see notebook §5* | *to be filled* |
| LLM prompt iteration (how many revisions, what each fixed) | *to be filled during implementation* | *anchored in per-iteration eval scores — see `notebooks/04_pipeline_eval.ipynb`* | *to be filled* |
| Confidence threshold for accepting LLM labels in the production run | *to be filled during implementation* | *anchored in precision-vs-coverage trade-off on eval set — see notebook §6* | *to be filled* |
| Eval set sampling strategy (stratification: by election × outlet origin × week) | *to be filled during implementation* | *anchored in expected-class-imbalance diagnostic — see notebook §5* | *to be filled* |

> Brand note: every choice above is an *educated* decision, not a convention. If you'd defend it differently, the diagnostic data is in the notebook — read it and tell me where I'm wrong.

## Findings

*To be filled during implementation. Each finding will be a falsifiable statement anchored in a specific number from the analysis (e.g., "international press allocated X% of frame-share to security in Kenya 2022, vs. Y% from African outlets, p < Z").*

## Limitations

*To be filled during implementation. Expected categories: GDELT coverage bias (it indexes what it indexes), English-only filter excludes francophone-only Senegal coverage from local outlets, LLM-as-classifier is an evaluated approximation not a ground truth, the frame taxonomy is itself a choice, and outlet-origin attribution treats "African" as a single block which it is not.*

## Visual style

This project uses **matplotlib + seaborn** for the stacked frame-distribution bars, the confusion matrices, and the per-election panel comparisons, because the deliverables are figures that need to read well in a LinkedIn or Twitter image preview and in a static recruiter-facing notebook. The flagship hero figure is a stacked bar of frame-mix (African vs. international press, per election) — a chart type that lives or dies on legible static rendering, not interactivity.

## How to reproduce

```bash
git clone <url>
cd 05-african-elections-frames

# Install with the NLP + LLM + viz extras
pip install -e ".[viz,nlp,llm]"

# NVIDIA API key is read from ../secrets.toml under [NVIDIA] API_KEY
# (see ../NVIDIA_API_request_sample.py for the call shape — OpenAI-compatible client,
#  base_url https://integrate.api.nvidia.com/v1)

# Run the pipeline notebooks in order
jupyter lab notebooks/02_main.ipynb
```

Full run time: ~X minutes for cached data; classification step depends on the number of articles (cost log committed to `data/processed/llm_cost.csv`).

## Files

- `notebooks/02_main.ipynb` — the analysis (start here)
- `notebooks/03_robustness.ipynb` — sensitivity checks for taxonomy granularity, confidence threshold, and dedup threshold
- `notebooks/04_pipeline_eval.ipynb` — precision/recall/F1 per frame, confusion matrix, prompt-iteration history, qualitative error analysis
- `src/elections_frames/data.py` — GDELT ingestion + caching + outlet provenance join
- `src/elections_frames/cleaning.py` — relevance filter + deduplication
- `src/elections_frames/classify.py` — NVIDIA NIM API wrapper (`deepseek-v4-pro` primary, `minimax-m2.7` hard-failure fallback) with structured output + cost logging
- `src/elections_frames/viz.py` — stacked-bar and confusion-matrix helpers (matplotlib + seaborn)
- `src/elections_frames/diagnostics.py` — diagnostic helpers used in decision blocks
- `data/external/outlets.csv` — hand-curated outlet provenance (African / International)
- `data/external/codebook.md` — frame taxonomy codebook
- `data/external/eval_set.parquet` — **hand-labeled by Muhanad** (gold set; never overwritten by pipeline output)
- `data/processed/articles_classified.parquet` — production-run LLM labels
- `data/processed/llm_cost.csv` — running cost log per pipeline run
- `figures/` — saved figures, including `hero.png` (committed)
- `tests/test_smoke.py` — minimal smoke tests

## Author

Muhanad — [LinkedIn](URL) · [Twitter](URL)
