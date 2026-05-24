"""Prompt v2 — enable and instruct abstention ("no clear frame").

Fix relative to v1
------------------
v1 scored micro-F1 0.441 / macro-F1 0.413 against the hand-labeled eval set, with
a single dominant pathology: **it never abstains.** v1's rules told the model to
return ``["process"]`` with low confidence when the input was too thin, and the
earlier schema could not express an empty label at all. But 135/250 eval rows
were hand-labeled with *no frame* (the GKG-metadata-only snippet is too thin, or
the article is not actually about the election). So v1 forced those 135 rows into
frames — 72 of them into ``process`` — and per-frame precision collapsed to ~0.31
(385 predicted frame-instances vs 155 in the gold).

v2 makes exactly one change: the model may, and should, **return an empty
``frames`` list with ``primary_frame: null`` when no single dimension is clearly
foregrounded** — because the snippet is too thin to tell, or because it is not
really about the election. This pairs with the schema relaxation
(``FrameClassification.frames`` may be empty). Everything else (the six frame
definitions, the foregrounding rule, JSON-only output) is held fixed so the score
delta isolates the abstention fix.

Iteration rule: each version is its own file
with a docstring noting what it fixed; the eval notebook reads them all.
"""

from __future__ import annotations

from elections_frames.prompts.v1 import FRAME_DEFINITIONS

PROMPT_VERSION = "v2"

# Few-shot examples: v1's three (security/process, economy, democracy) plus a
# fourth that demonstrates ABSTENTION on a thin / no-frame snippet — the behavior
# v1 lacked. Example 4 is written in the metadata shape the production inputs take
# ("Title: ... | Names: ... | Themes: ...") so the abstain case is recognizable.
FEW_SHOT_EXAMPLES = """\
EXAMPLE 1
INPUT: "Polling stations in three northern states opened late after armed groups attacked \
voters in transit overnight. The electoral commission said two officials were killed and \
materials destroyed; voting was suspended in 14 wards."
OUTPUT:
{
  "frames": ["security", "process"],
  "primary_frame": "security",
  "confidence": 0.9,
  "rationale": "The lead foregrounds violence against voters and election workers (security); the procedural disruption — suspended voting, destroyed materials — is also foregrounded but downstream of the violence itself."
}

EXAMPLE 2
INPUT: "With the currency at a record low and inflation in double digits, candidates spent \
the final week of the campaign promising subsidy reform and currency stabilization. \
Opposition rallies drew large crowds in cities hit hardest by the cost-of-living crisis."
OUTPUT:
{
  "frames": ["economy"],
  "primary_frame": "economy",
  "confidence": 0.95,
  "rationale": "The article frames the vote almost entirely through macroeconomic conditions and economic-policy promises; no other dimension is foregrounded."
}

EXAMPLE 3
INPUT: "The constitutional court will rule next week on whether the incumbent's removal of \
the chief justice last year violated separation-of-powers norms. Critics say it is part of \
a wider pattern of executive overreach; the ruling party says the reforms restored \
democratic accountability."
OUTPUT:
{
  "frames": ["democracy"],
  "primary_frame": "democracy",
  "confidence": 0.85,
  "rationale": "The vote is framed through a systemic question about institutional health and democratic norms (separation of powers, executive overreach), not through the mechanics of this specific election."
}

EXAMPLE 4
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
