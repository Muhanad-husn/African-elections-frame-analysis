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

A four-stage pipeline. **(1) Ingestion** pulls GDELT GKG records for the ±30-day window around each of the four elections. **(2) Cleaning** filters to election-relevant English-language articles, deduplicates, and attributes each outlet to "African" or "International" using a hand-curated provenance list. **(3) Classification** uses the OpenRouter API (`deepseek/deepseek-v4-flash` primary, `minimax/minimax-m2.7` hard-failure fallback) with a structured-output prompt to assign one or more frames per article from a 6-frame taxonomy (security / economy / democracy / identity / process / corruption). **(4) Evaluation** compares LLM labels against a hand-labeled set of 200–300 stratified articles, computing precision/recall/F1 per frame and a confusion matrix. Only after the eval passes a documented quality bar do we treat the production-run labels as analytically usable. The strongest critique of this design is that the frame taxonomy is itself a choice — different taxonomies would yield different distributions. The taxonomy is documented in a codebook (`data/external/codebook.md`) and its granularity is one of the documented decisions below.

## Methodological decisions

Each major data-processing decision was made by **diagnostic first, choice second**. The table below is an at-a-glance summary; the full five-part rationale (problem / diagnostic / options / decision + rationale / sensitivity) lives inline in `notebooks/02_main.ipynb`.

| Decision | Chose | Why (anchored in diagnostic) | Sensitivity |
|----------|-------|------------------------------|-------------|
| Outlet origin attribution (African vs. International; edge cases like BBC Africa service, Al Jazeera English Africa desk) | *to be filled during implementation* | *anchored in diagnostic — see notebook §3* | *to be filled* |
| Article-relevance filter (theme-tag threshold vs. keyword filter vs. hybrid) | *to be filled during implementation* | *anchored in diagnostic — see notebook §3* | *to be filled* |
| Deduplication strategy (URL canonicalization + near-duplicate text-similarity threshold) | *to be filled during implementation* | *anchored in diagnostic — see notebook §4* | **Not load-bearing.** Re-running dedup at Jaccard 0.7/0.9 vs. the 0.8 production choice changes the kept-row count ≤ 2% on the probe, and the marginal "swing" rows are origin-balanced (~73% International / 27% African, matching the corpus ~74/26) — so they cannot move a between-origin gap. `03_robustness.ipynb` §3. |
| Frame taxonomy granularity (5 vs. 6 vs. 7 frames; boundary cases) | *to be filled during implementation* | *anchored in eval-set boundary diagnostics — see notebook §5* | **Load-bearing.** Merging *democracy* + *process* → *governance* (5-frame collapse) shrinks the headline gap from +11.9/−10.4 pp to **+1.5 pp**: the two presses give governance near-identical *total* attention and diverge only in *how they split it*. The 6-frame distinction is kept on substantive grounds (the split *is* the finding). `03_robustness.ipynb` §1. |
| LLM prompt iteration (how many revisions, what each fixed) | *to be filled during implementation* | *anchored in per-iteration eval scores — see `notebooks/04_pipeline_eval.ipynb`* | *to be filled* |
| Confidence threshold for accepting LLM labels in the production run | *to be filled during implementation* | *anchored in precision-vs-coverage trade-off on eval set — see notebook §6* | **Robust.** Re-running the headline accepted-only at 0.70/0.75/0.80 holds the process gap at +10.5…+12.2 pp and the democracy gap at −11.2…−9.0 pp; tightening the gate lowers absolute levels (it is not frame-neutral) but is origin-neutral in the difference. `03_robustness.ipynb` §2. |
| Eval set sampling strategy (stratification: by election × outlet origin × week) | *to be filled during implementation* | *anchored in expected-class-imbalance diagnostic — see notebook §5* | *to be filled* |

> Brand note: every choice above is an *educated* decision, not a convention. If you'd defend it differently, the diagnostic data is in the notebook — read it and tell me where I'm wrong.

## Findings

*Draft (Session 7) — descriptive, not causal; bounded by the limitations below. Full analysis with figures in `notebooks/02_main.ipynb` §7–§9. Numbers are over **framed** articles (those the classifier foregrounded a frame for), pooled across the four elections unless noted.*

1. **African outlets foreground the *mechanics* of elections; international outlets foreground *democratic stakes*.** African coverage gives *process* (registration, counting, results, court challenges) **39.0%** of framed articles vs. **27.1%** international (**+11.9 pp**); international coverage gives *democracy* (institutional health, backsliding/consolidation, civil liberties) **26.9%** vs. **16.5%** African (**+10.4 pp**). These are the two largest origin gaps; every other frame differs by ≤ 3 pp.
2. **That split is directionally stable across elections — except Senegal.** Process is higher in African coverage in 3 of 4 elections (Nigeria +4.5, Kenya +9.5, South Africa +7.5 pp); democracy is higher in international coverage in the same 3. Senegal 2024 inverts it (both origins ~51–61% *democracy*) because the postponement/constitutional crisis was the story.
3. **International coverage is far more often too thin to frame.** **61.9%** of international rows abstain (no foregrounded frame) vs. **44.6%** of African rows — international coverage skews to short wire/aggregator briefs.
4. **Coverage pivots from issues to mechanics around vote day.** *Process* rises **23.4% → 36.7%** from the pre-period (−30…−8 days) to the post-period (+8…+30) while *economy* falls **20.7% → 10.9%**.
5. **"African framing" is not monolithic.** Within the African block, Nigerian (process **45.7%**) and Kenyan (process **50.1%**) outlets are process-heavy while South African outlets are democracy/economy-balanced (democracy **27.4%**, economy **18.9%**).

## Limitations

- **GDELT coverage bias.** The corpus is whatever GDELT crawls — English-language, publicly indexed sources. It is not a census of coverage.
- **English-only filter excludes the francophone Senegalese press.** Senegal's African sample is just 312 framed rows (~18 from `.sn` domains); local Wolof/French framing is largely absent, reachable only via international wires. The Senegal panel is read with caution.
- **GKG-metadata-only input caps classifier precision** (Decision Log #1). No article bodies were scraped — the model frames a metadata snippet. 54% of hand-labeled eval rows were too thin to frame, and v3's eval micro-F1 is 0.552, so the labels are an *evaluated approximation*, not ground truth. The 0.75 confidence gate is reported as a flag, not applied to the headline, because it is not frame-neutral.
- **Abstention is informative but shrinks the effective sample.** The framed-only mix is computed on ~38–45% of rows; the abstaining majority carries no frame by construction.
- **The 6-frame taxonomy is one defensible cut.** The democracy↔process seam is the dominant residual confusion (merging the two lifts eval micro-F1 0.552→0.598); it is stress-tested in `03_robustness.ipynb`, not assumed away.
- **"African" is a block at the headline level.** Intra-African variation is real (notebook §7.4) and surfaced separately.
- **Descriptive, not causal.** "Outlet origin X foregrounds frame Y at Z%" — no claim about *why*, or about coverage quality/accuracy.

## Visual style

This project uses **matplotlib + seaborn** for the stacked frame-distribution bars, the confusion matrices, and the per-election panel comparisons, because the deliverables are figures that need to read well in a LinkedIn or Twitter image preview and in a static recruiter-facing notebook. The flagship hero figure is a stacked bar of frame-mix (African vs. international press, per election) — a chart type that lives or dies on legible static rendering, not interactivity.

## How to reproduce

```bash
git clone <url>
cd 05-african-elections-frames

# Install with the NLP + LLM + viz extras
pip install -e ".[viz,nlp,llm]"

# OpenRouter API key is read from ../secrets.toml under [OPENROUTER] OPENROUTER_API_KEY
# (OpenAI-compatible client, base_url https://openrouter.ai/api/v1).
# NVIDIA NIM is retained as a switchable provider (provider="nvidia") but inactive
# by default — see classify.py for why (free-tier latency/error rate).

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
- `src/elections_frames/classify.py` — OpenRouter API wrapper (`deepseek-v4-flash` primary, `minimax-m2.7` hard-failure fallback; NVIDIA NIM switchable but inactive) with structured output + cost logging
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
