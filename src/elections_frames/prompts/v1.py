"""Prompt v1 — initial draft, no eval data yet.

Codebook-grounded system prompt with the 6-frame taxonomy from
``data/external/codebook.md`` v0. Few-shot examples are synthesized to
illustrate each frame; they do **not** quote any specific real article and
are explicitly NOT drawn from ``data/external/eval_set.parquet`` (which is
hand-labeled and must not leak into prompt design).

Iteration rule (per IMPLEMENTATION_PLAN.md Session 5): each subsequent
version lives in its own file (``v2.py``, ``v3.py``, ...) with a docstring
noting what it fixed relative to the previous version. ``eval_pipeline.ipynb``
reads them all to produce the per-version score table.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

# Codebook excerpt — kept in sync with data/external/codebook.md §"The six frames".
# Reproduced inline so prompt iteration is self-contained (no notebook-time codebook
# changes silently affect the prompt; later versions explicitly opt in).
FRAME_DEFINITIONS = """\
1. **security** — Election framed in terms of physical safety, public order, violence, \
intimidation, terrorism, armed conflict, militia/insurgent activity, policing, or the \
credibility of state coercive capacity. Includes both election-period violence and \
background conflict that shapes the vote.

2. **economy** — Election framed in terms of macroeconomic conditions, fiscal policy, \
debt, currency, inflation, employment, growth, sector policy, trade, or business \
confidence. Includes "cost-of-living election" framings.

3. **democracy** — Election framed in terms of democratic norms, institutional health, \
constitutionalism, civil liberties, press freedom, term limits, rule of law, opposition \
space, or backsliding/consolidation narratives. Distinct from process: democracy is about \
whether the polity is democratic; process is about whether *this* vote was clean.

4. **identity** — Election framed in terms of ethnicity, religion, region, language, \
generation, gender, or other ascriptive group identities and the politics organized \
around them. Includes coalition arithmetic explained through identity blocs.

5. **process** — Election framed in terms of the mechanics and integrity of this specific \
vote: registration, logistics, polling-day operations, counting, results transmission, \
court challenges, observer reports, specific irregularities. Procedural, not systemic.

6. **corruption** — Election framed in terms of graft, kleptocracy, illicit enrichment, \
patronage networks, vote-buying, public-sector capture, or accountability deficits framed \
as endemic corruption rather than as discrete procedural irregularities (which would be process).\
"""

# Synthesized few-shot examples illustrating each frame and multi-label assignment.
# These are PEDAGOGICAL ILLUSTRATIONS, not excerpts of any specific article, and they
# are not drawn from eval_set.parquet.
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
  "frames": ["frame_name", ...],          // 1-3 frames, in order of prominence
  "primary_frame": "frame_name",          // the single most foregrounded frame
  "confidence": 0.0-1.0,                  // your overall confidence in this classification
  "rationale": "string"                   // 2-3 sentences, anchored in what the input actually says
}}

Allowed frame_name values: "security", "economy", "democracy", "identity", "process", "corruption". \
Lowercase only. No other values.

# Rules

- Multi-label is allowed but rare. Most articles foreground 1-2 frames. Three is unusual; \
four or more means you are probably reading the article too generously — pick the dominant ones.
- Frames mentioned only in passing do NOT count. A frame must be foregrounded.
- If the input is too thin to support any confident frame assignment (e.g., metadata-only with \
no actual framing signal), return ["process"] with low confidence and say so in the rationale.
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
