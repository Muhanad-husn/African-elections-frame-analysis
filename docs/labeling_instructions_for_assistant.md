# Labeling instructions — for the assistant

You are hand-labeling 250 news articles for a frame-analysis project. Your output is the **gold standard** the project's automated classifier will be evaluated against.

---

## The one rule that matters most

If you're unsure about an article, **use the Notes box** in the widget and assign your best guess. Uncertainty is data, not a problem.

---

## How to start 

1. Open a terminal in the project folder: `D:\github-ds-portfolio\05-african-elections-frames`
2. Run:
   ```
   conda activate portfolio
   jupyter lab
   ```
3. In JupyterLab, open `notebooks/eval_labeling.ipynb`
4. From the menu: **Run → Run All Cells**
5. The labeling form appears under the last cell. Start labeling.

If conda or JupyterLab isn't installed, or any step errors out, **stop and message Muhanad** — don't try to fix it yourself, the environment was set up specifically for this work.

---

## What you're labeling

Each article shows you:
- The **outlet** (e.g., `bbc.com`, `nation.africa`) and which **election** it relates to (Nigeria 2023, Kenya 2022, Senegal 2024, or South Africa 2024)
- The **URL** (clickable — feel free to skim the original article if the snippet is too thin)
- A **text snippet** built from URL keywords, named entities, and theme tags

You decide which of the **6 frames** are *foregrounded* in the article. Multi-label is fine — most articles foreground 1–2 frames.

| Frame | Means |
|---|---|
| **security** | Physical safety, violence, intimidation, terrorism, conflict, policing |
| **economy** | Macroeconomy, debt, inflation, employment, growth, business |
| **democracy** | Democratic norms, institutional health, constitutionalism, press freedom, backsliding |
| **identity** | Ethnicity, religion, region, language, generation, gender |
| **process** | Mechanics of *this* vote: registration, logistics, counting, court challenges, alleged rigging |
| **corruption** | Graft, kleptocracy, patronage, vote-buying, endemic accountability deficits |

The widget shows this table inline so you don't need to memorize it. The full codebook (with boundary-case heuristics) is `data/external/codebook.md`.

---

## Three buttons, two outcomes

- **Save & Next** — tick at least one frame, then press this. Your label is recorded and the next article appears.
- **Skip (no label)** — press this if the snippet is too thin to tell, the URL is 404, or the article isn't actually about an African election (the relevance filter occasionally lets unrelated articles slip through). Skipped articles are marked `too_thin=True` and excluded from the precision/recall calculation.
- **Notes (optional textbox)** — leave a note for any article where you hesitated, or where you think the codebook should be sharper. The notes are read at the next stage of the project to inform whether the 6 frames need to be merged/split.

---

## Quality bar

- **Aim for 5–15 seconds of judgment per article.** This is fast-skim labeling, not deep analysis. The snippet plus a glance at the URL is enough for most.
- **Spend longer on edge cases** — articles that feel ambiguous between two frames (security/process, democracy/process, identity/corruption, economy/corruption). Pick both labels if the article genuinely foregrounds both.
- **Don't tick all 6 frames.** If you want to, the snippet is too generic — press Skip instead.
- **Be consistent.** If you'd label an article one way at row 50, label a similar one the same way at row 200.

---

## When you're done

1. The widget will display "All articles labeled. Thank you!"
2. Message Muhanad and let them know it's ready for review.
3. **Do not edit `data/external/eval_set.parquet` by hand**, and **do not push to the repo yourself** — Muhanad will commit the labeled file.

---

## If something goes wrong

- **Widget doesn't appear**: scroll down past the last cell, or refresh the JupyterLab page and re-run all cells.
- **You picked the wrong label and need to go back**: the widget has no back button by design. Note the article number (shown at the top: "Article N/250") and the correction in a separate text file; Muhanad will fix it post-hoc.
- **You realize you've been mis-applying the codebook**: stop, message Muhanad. Better to recalibrate now than to power through 250 inconsistent labels.
- **Any error / crash / "this looks broken"**: stop, message Muhanad with a screenshot of the error.

---

## Why this matters (one paragraph for context)

The project measures whether international and African news outlets frame African elections differently — across dimensions like security, economy, democracy, identity, process, and corruption. The LLM classifier guesses these frames automatically; your 250 hand-labels are the independent ground truth that tells us whether the classifier is any good. Every label you assign is one data point in a precision/recall calculation that gates whether the project's headline finding can be trusted. **You labeling carefully and consistently is more valuable than labeling fast.**

Reference docs (only read if you want more depth — they're not required):
- `data/external/codebook.md` — full frame definitions + boundary-case heuristics
- `docs/labeling_handoff.md` — the methodology context

Questions: message Muhanad. Don't ask an AI.
