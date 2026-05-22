"""Prompt v4 — sharpen the democracy <-> process boundary.

Fix relative to v3
------------------
v3 (metadata-format examples) lifted micro-F1 to 0.552 and recovered process
recall, but it over-predicts **democracy**: 44 predicted vs 29 in the gold, 21
false positives — now the single largest error source. Inspecting those 21 FPs,
the pattern is the codebook's #1 boundary case (``democracy <-> process``): the
model tags result disputes, court challenges, and electoral-commission conduct
about *this specific vote* as ``democracy`` (a systemic-norms reading), whereas
the human labeled them ``process`` (the integrity/mechanics of this count). E.g.
"iebc quorum did chebukati err announce presidential results", "court gives inec
nod reconfigure bvas" — both gold ``process``, both tagged ``democracy`` by v3.

v4 makes exactly one change: a **disambiguation rule** (plus one contrastive
example) that routes court challenges / result disputes / electoral-commission
conduct / observer verdicts about *this* vote to ``process``, and reserves
``democracy`` for systemic questions (term limits, press freedom, constitutional
order, backsliding) that transcend the specific count. Everything else is
identical to v3, so the score delta isolates the boundary fix.

This is also a probe for the Session-5 taxonomy-granularity decision: if a sharp
verbal rule cannot separate democracy from process, that is evidence the seam is
inherent to the taxonomy rather than a prompt defect.
"""

from __future__ import annotations

from elections_frames.prompts.v1 import FRAME_DEFINITIONS

PROMPT_VERSION = "v4"

# v3's metadata-format examples, plus Example 6: a systemic-democracy snippet that
# must NOT collapse to process, to anchor the boundary from the democracy side.
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
  "confidence": 0.85,
  "rationale": "A court challenge to the tally and the electoral commission's defense of THIS count foreground the integrity and mechanics of this specific vote — process. There is no systemic-norms claim, so this is not democracy."
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
  "rationale": "The snippet is generic post-election political housekeeping (a caucus meeting). No single dimension is clearly foregrounded, so no frame is assigned."
}

EXAMPLE 6
INPUT: "Title: rights groups warn term limit removal press crackdown ahead of vote | Names: Constitutional Court, Human Rights Watch, Press Union | Themes: ELECTION, CONSTITUTION, FREEDOM_OF_THE_PRESS, DEMOCRACY, ARREST"
OUTPUT:
{
  "frames": ["democracy"],
  "primary_frame": "democracy",
  "confidence": 0.85,
  "rationale": "Term-limit removal and a press crackdown foreground systemic democratic norms and institutional health that transcend this particular count — democracy, not the procedural conduct of the vote itself."
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
- **democracy vs process.** Anything about *this specific vote* — court challenges to the result, \
result disputes, vote counting and transmission, the electoral commission's conduct, observer \
verdicts, registration and polling logistics, alleged irregularities — is **process**. Reserve \
**democracy** for systemic questions that transcend this count: term limits, constitutional order, \
press freedom, civil liberties, opposition space, rule-of-law backsliding or consolidation. \
A disputed tally is process; a dismantled term limit is democracy.
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
