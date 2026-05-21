"""Versioned classifier prompts.

Each prompt iteration lives in its own module (``v1.py``, ``v2.py``, ...)
with a docstring noting what it fixed relative to the previous version.
The per-version narrative is itself a deliverable (see Session 5 in
``IMPLEMENTATION_PLAN.md``).

Lookup helper: ``get(version)`` returns the prompt module so ``classify.py``
can drive any version without hard-coding imports.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def get(version: str) -> ModuleType:
    """Return the prompt module for ``version`` (e.g. ``"v1"``)."""
    return importlib.import_module(f".{version}", package=__name__)

