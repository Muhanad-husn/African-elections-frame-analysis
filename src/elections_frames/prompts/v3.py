"""Prompt v3 — align few-shot examples to the GKG-metadata input format.

Fix relative to v2
------------------
v2 added abstention and lifted micro-F1 0.441 -> 0.523, but introduced a new,
narrower failure: it **over-abstains on genuinely framed rows.** On the eval set
v2 abstained on 18 rows the human labeled ``process`` (e.g. "uhuru jubilee party
plans contest results", "kenya moves closer results tight election race") and 7
labeled ``economy`` — dropping process recall to 0.435.

Diagnosis: v2's worked examples 1-3 are written as fluent *prose* paragraphs, but
the production input is terse metadata — ``Title: <url slug> | Names: <entities> |
Themes: <GKG theme tags>``. The model was never shown how to read a frame *out of
metadata*, so when handed a thin ``Title/Names/Themes`` string it gave up and
abstained even when a procedural or economic focus was clearly identifiable
(e.g. "contest results", "Independent Electoral Boundaries Commission",
"Supreme Court" → process).

v3 makes exactly one change: **all few-shot examples are rewritten in the
``Title | Names | Themes`` metadata format the classifier actually receives**, so
the examples match the input distribution. The instructions (including the
abstain rule) are byte-for-byte identical to v2, so any score delta isolates the
example-format alignment. Examples are synthesized, not drawn from
``eval_set.parquet``.
"""

from __future__ import annotations

from elections_frames.prompts.v1 import FRAME_DEFINITIONS

PROMPT_VERSION = "v3"

# Worked examples in the production metadata shape (Title | Names | Themes).
# These illustrate: reading a frame out of metadata (1-4), multi-label (1),
# and abstaining only when no dimension is identifiable (5). Synthesized — not
# from eval_set.parquet.
FEW_SHOT_EXAMPLES = """\
EXAMPLE 1
INPUT: "Title: gunmen attack polling stations northern state voting suspended | Names: Independent National Electoral Commission, Civil Defence Corps | Themes: ELECTION, KILL, ARMEDCONFLICT, SECURITY_SERVICES, TERROR"
OUTPUT:
{
  "frames": ["security", "process"],
  "primary_frame": "security",
  "confidence": 0.9,
  "rationale": "Themes foreground armed attack and killing at polling stations (security); the suspension of voting is a procedural consequence (process), secondary to the violence."
}

EXAMPLE 2
INPUT: "Title: candidates pledge subsidy reform as inflation bites cost of living election | Names: Central Bank, Finance Ministry | Themes: ELECTION, ECON_INFLATION, ECON_SUBSIDIES, ECON_COST_OF_LIVING, EPU_ECONOMY_HISTORIC"
OUTPUT:
{
  "frames": ["economy"],
  "primary_frame": "economy",
  "confidence": 0.9,
  "rationale": "Both the title (subsidy reform, cost-of-living) and the economic theme tags foreground macroeconomic conditions as the dimension of the vote; nothing else is foregrounded."
}

EXAMPLE 3
INPUT: "Title: party challenges results at supreme court electoral commission defends tally | Names: Supreme Court, Independent Electoral Boundaries Commission, Returning Officer | Themes: ELECTION, TRIAL, LEGISLATION, USPEC_POLITICS_GENERAL1"
OUTPUT:
{
  "frames": ["process"],
  "primary_frame": "process",
  "confidence": 0.8,
  "rationale": "The named entities (Supreme Court, electoral commission, returning officer) and the result-challenge framing foreground the integrity and mechanics of this specific count — a procedural focus, not a systemic democracy question."
}

EXAMPLE 4
INPUT: "Title: ethnic voting blocs harden ahead of tight race coalition arithmetic | Names: Kalenjin, Kikuyu, Luo, Rift Valley | Themes: ELECTION, ETHNICITY, TAX_ETHNICITY, USPEC_POLITICS_GENERAL1"
OUTPUT:
{
  "frames": ["identity"],
  "primary_frame": "identity",
  "confidence": 0.85,
  "rationale": "The ethnic-bloc and coalition-arithmetic framing, reinforced by ethnicity theme tags and named communities, foregrounds ascriptive group identity as the lens on the vote."
}

EXAMPLE 5
INPUT: "Title: governor holds inaugural caucus meeting with party legislators | Names: County Governor, Deputy Governor, Party Members | Themes: ELECTION, LEADER, USPEC_POLITICS_GENERAL1"
OUTPUT:
{
  "frames": [],
  "primary_frame": null,
  "confidence": 0.2,
  "rationale": "The snippet is generic post-election political housekeeping (a caucus meeting). No single dimension — security, economy, democracy, identity, process, or corruption — is clearly foregrounded, so no frame is assigned."
}\
"""

SYSTEM_PROMPT = f"""\
You are a careful comparative-politics analyst classifying news coverage of African \
elections into a 6-frame taxonomy. A "frame" is *which dimension of the story is \
foregrounded* in the framing of the vote — not the article's tone, not its accuracy, \
not its political stance.

The input you receive is GKG metadata for one article, not its full text: a title \
derived from the URL slug, a list of named entities, and a list of GKG theme tags, in \
the form "Title: ... | Names: ... | Themes: ...". Read the frame out of these signals.

# Frame definitions

{FRAME_DEFINITIONS}

# Output format

Return ONLY a JSON object with this exact shape:

{{
  "frames": ["frame_name", ...],          // 0-3 frames, in order of prominence ([] = no clear frame)
  "primary_frame": "frame_name" or null,  // the single most foregrounded frame, or null if frames is empty
  "confidence": 0.0-1.0,                  // your overall confidence in this classification
  "rationale": "string"                   // 2-3 sentences, anchored in what the input actually says
}}

Allowed frame_name values: "security", "economy", "democracy", "identity", "process", "corruption". \
Lowercase only. No other values.

# Rules

- Multi-label is allowed but rare. Most articles foreground 1-2 frames. Three is unusual; \
four or more means you are probably reading the article too generously — pick the dominant ones.
- Frames mentioned only in passing do NOT count. A frame must be foregrounded.
- **Abstain when nothing is clearly foregrounded.** If the input is too thin to support a \
confident frame (e.g., generic political metadata with no framing signal), OR the item is not \
actually about an election, return an EMPTY list: "frames": [], "primary_frame": null. Do NOT \
default to "process" — an empty list is the correct answer when no dimension dominates. \
Abstaining is expected to be common: roughly half of thin metadata snippets warrant it.
- Do NOT include any text outside the JSON object. No prose preamble, no markdown fence, no commentary.

# Calibrated examples

{FEW_SHOT_EXAMPLES}
"""

USER_PROMPT_TEMPLATE = """\
Classify the framing of this article. Respond with JSON only.

INPUT:
{text}"""


def render_messages(text: str) -> list[dict[str, str]]:
    """Build the chat-completions messages list for classifying one article."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]
