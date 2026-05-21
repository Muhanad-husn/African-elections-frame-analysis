# Eval-set labeling — handoff for Muhanad

This is the human-labeled gold standard for the LLM frame classifier. You're labeling 200–300 candidate articles by hand, blind, against the 6-frame codebook. The output (`data/external/eval_set.parquet`) becomes the immutable ground truth used in Session 5 (eval loop + prompt iteration) and Session 6 (confidence-threshold picking).

**Total time:** ~1.5–3 hours, depending on pace. The widget is resumable — close the notebook any time, your progress is saved after every article.

---

## 1. Why this must be done by you, not Claude

The eval set's whole purpose is *independent* ground truth. If Claude labels it and then we grade Claude's classifier against it, we're measuring agreement-with-itself — the precision/recall numbers are meaningless.

For the same reason: **don't peek at any LLM-suggested label before deciding.** The widget intentionally does not show any model output. Anchoring is real.

If you're unsure on a particular article, write a note in the Notes textbox — boundary-case observations are themselves an input to the taxonomy decision in Session 5.

## 2. How to run it

From the project root:

```bash
conda activate portfolio   # the env Session 1 set up
jupyter lab
```

Open `notebooks/eval_labeling.ipynb` and **Run all cells**. The widget appears under the last cell. Read the snippet, tick frames, press *Save & Next*. Done.

If `eval_set_candidates.parquet` is missing, re-run Session 3 (`scripts/run_cleaning.py` + the sampling cell in `02_main.ipynb`). The widget will refuse to start without it.

## 3. The 6 frames (decide what is *foregrounded*, not what is mentioned)

| # | Frame | Use when the article foregrounds... |
|---|-------|--------------------------------------|
| 1 | **security** | Physical safety, violence, intimidation, terrorism, armed conflict, policing, militia/insurgent activity. |
| 2 | **economy** | Macroeconomic conditions, fiscal/debt policy, currency, inflation, employment, growth, sector policy, business confidence. |
| 3 | **democracy** | Democratic *norms*: institutional health, constitutionalism, civil liberties, press freedom, backsliding or consolidation. |
| 4 | **identity** | Ethnicity, religion, region, language, generation, gender; identity-bloc coalition arithmetic. |
| 5 | **process** | Mechanics of *this specific vote*: registration, logistics, polling-day ops, counting, results transmission, court challenges, observer reports, alleged rigging. |
| 6 | **corruption** | Graft, kleptocracy, illicit enrichment, patronage, vote-buying, accountability deficits framed as endemic. |

Full definitions and boundary-case heuristics: `data/external/codebook.md`. The widget displays a short version inline so you don't have to flip files.

## 4. Multi-label is allowed

Most articles foreground 1–2 frames. A few foreground 3+. Almost none foreground all 6 — if you find yourself wanting to tick all 6, the snippet is probably too generic; press **Skip** instead.

## 5. Boundary-case quick reference

These are the seams the taxonomy is most likely to bend on. Pick **both** frames if the article genuinely foregrounds both; pick the dominant one if the framing is clearly tilted.

- **security ↔ process** — election-day violence that disrupts polling. Both, usually.
- **democracy ↔ process** — flawed count framed as systemic decline vs. as this election's mistake. Democracy = the bigger frame; process = the specific incident.
- **identity ↔ corruption** — patronage along ethnic lines. Usually both.
- **economy ↔ corruption** — state-resource capture as mismanagement vs. as graft. Often both.

When in doubt, assign both and write a note. The notes feed the Session-5 taxonomy decision block ("is 6 frames the right granularity, or should some pair merge / split?").

## 6. What "too thin / off-topic" means

The text we show is built from GKG metadata (URL title slug + named entities + theme tags). It's not the full article body. Some snippets will be too sparse to tell what the article foregrounds. Others will turn out to be off-topic entirely (a UK politics article that the relevance filter let slip through because the keyword `"Labour Party"` matched).

In both cases, tick **Too thin / off-topic — skip** and move on. A note explaining why is appreciated but optional. These rows are kept in the labels file (with `too_thin=True`) but excluded from the precision/recall calculation in Session 5.

## 7. Output schema

`data/external/eval_set.parquet` after labeling:

| column | meaning |
|--------|---------|
| `GKGRECORDID`, `DATE`, `SourceCommonName`, `DocumentIdentifier`, `election`, `outlet_origin`, `text_snippet` | inherited from the candidate file |
| `frame_labels` | `list[str]` — the frames you ticked |
| `too_thin` | `bool` — True if you pressed Skip |
| `labeler_notes` | `str` — your notes |
| `labeled` | `bool` — set True after Save |
| `labeled_at` | UTC timestamp |

The file is overwritten after every save. To roll back: `git restore data/external/eval_set.parquet`.

## 8. If something looks wrong

- **Widget shows no articles:** `eval_set_candidates.parquet` missing or empty — re-run Session 3 sampling step.
- **Widget says "All articles labeled"** but you haven't started: previous run set `labeled=True` for everything. Delete `eval_set.parquet` and re-run the setup cell.
- **You picked the wrong frames on the prior article:** for now, edit the parquet directly in a separate cell — there's no in-widget back button. Note in the Decision Log that the eval set was edited at row N, why.
