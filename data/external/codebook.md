# Frame taxonomy codebook — v0

Six frames, one label can apply to multiple frames (the classifier returns a
list, not a single label). This is **v0** — the taxonomy will be revisited
after labeling the first ~50 articles, when boundary cases observed in
practice can inform whether the right granularity is 5, 6, or 7 frames
(see the taxonomy decision block in
`notebooks/04_pipeline_eval.ipynb`).

## The six frames

### 1. Security
Election framed in terms of physical safety, public order, violence,
intimidation, terrorism, armed conflict, militia or insurgent activity,
policing, or the credibility of state coercive capacity. Includes both
*election-period* violence and references to background conflict that
shape the vote.

### 2. Economy
Election framed in terms of macroeconomic conditions, fiscal policy, debt,
currency, inflation, employment, growth, sector policy (energy, agriculture,
mining), trade, or business confidence. Includes "the cost-of-living election"
type framings.

### 3. Democracy
Election framed in terms of democratic norms, institutional health,
constitutionalism, civil liberties, press freedom, term limits, the rule of
law, opposition space, or backsliding/consolidation narratives. Distinct
from process (#5) — democracy is about whether the polity is democratic;
process is about whether *this election* was conducted cleanly.

### 4. Identity
Election framed in terms of ethnicity, religion, region, language, generation,
gender, or other ascriptive group identities and the politics organized
around them. Includes coalition arithmetic explained through identity
blocs, and identity-based mobilization or backlash.

### 5. Process
Election framed in terms of the mechanics and integrity of this specific
vote: registration, logistics, polling-day operations, counting, results
transmission, court challenges, observer reports, allegations of rigging,
specific irregularities. Procedural rather than systemic.

### 6. Corruption
Election framed in terms of graft, kleptocracy, illicit enrichment,
patronage networks, vote-buying, public-sector capture, or accountability
deficits framed as endemic corruption rather than as discrete procedural
irregularities (which would be process).

## Boundary cases to watch (informs v1 revision)

These are the seams where the taxonomy is likely to bend on contact with
labeled data. Note them as they come up during labeling:

- **security ↔ process** — election-day violence that disrupts polling
  (e.g., booth captures): is the foregrounded dimension physical safety
  or procedural integrity?
- **democracy ↔ process** — a flawed count where the framing emphasizes
  systemic democratic decline rather than the specific count.
- **identity ↔ corruption** — patronage along ethnic lines.
- **economy ↔ corruption** — capture of state resources framed as
  economic mismanagement vs. as graft.

The handling rule (multi-label) means we can assign both labels and revisit
the taxonomy after observing how often co-assignment happens.

## Labeling instructions (for the eval set)

- Read the snippet, decide which frames are *foregrounded* (not merely
  mentioned in passing).
- Multi-label is allowed — most articles foreground 1–2 frames; very few
  foreground 3+. If you find yourself wanting to assign all six, the
  snippet is probably too generic to label — flag it.
- If a snippet is ambiguous between two frames and you cannot pick, assign
  both and add a note. The notes are an input to the taxonomy revision.
- **Do not** look at any LLM-suggested label before assigning your own.
  Anchoring bias is real — labeling must be done blind.
