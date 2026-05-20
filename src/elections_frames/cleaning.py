"""Article cleaning: English-language filter, relevance filter, deduplication.

Filled in during Session 3 (see ``IMPLEMENTATION_PLAN.md``). Planned surface:

- ``filter_english(df, text_col)`` — drop non-English rows using ``langdetect``.
- ``filter_relevant(df, election)`` — keep articles that are about the
  named election (theme-tag and/or keyword based; the exact rule is the
  output of a five-part decision block in ``02_main.ipynb``).
- ``deduplicate(df, text_col, jaccard_threshold)`` — URL canonicalization
  + MinHash near-duplicate detection. Threshold selected via decision block.

All three steps emit diagnostic counts so notebook cells can show before/after.
"""

from __future__ import annotations
