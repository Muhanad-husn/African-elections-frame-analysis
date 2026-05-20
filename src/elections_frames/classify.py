"""NVIDIA NIM classifier wrapper with structured output + cost logging.

Filled in during Session 4 (see ``IMPLEMENTATION_PLAN.md``). Planned surface:

- ``classify_article(text, prompt_version)`` — call NVIDIA NIM
  (``deepseek-ai/deepseek-v4-pro`` primary, ``minimaxai/minimax-m2.7``
  hard-failure fallback) via the OpenAI-compatible client at
  ``https://integrate.api.nvidia.com/v1``. Parse structured output
  ``{frames: list[str], confidence: float, rationale: str}``, log input/output
  tokens + cost to ``data/processed/llm_cost.csv``, retry on transient errors.

API key is read from ``../secrets.toml`` (monorepo parent) under
``[NVIDIA] API_KEY`` using stdlib ``tomllib`` (Python 3.11+).

Cost is a first-class success metric (Principle 2): every run appends a row
to the cost log; notebooks surface a running total.
"""

from __future__ import annotations
