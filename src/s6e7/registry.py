"""Model factories: a dict of callables keyed by name.

Builders live in ``s6e7.models.*`` and self-register via the decorator; ``build`` imports
that package lazily, which is what fills the dict without an import cycle. A builder gets
the experiment's params dict and returns a *fresh, unfitted* estimator — cv.run calls it
once per fold so no state leaks between folds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from s6e7.protocols import Classifier

Builder = Callable[[dict[str, Any]], Classifier]

BUILDERS: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator: file the builder under `name`. Duplicate names are a bug, not an update."""

    def deco(fn: Builder) -> Builder:
        if name in BUILDERS:
            msg = f"model {name!r} is already registered"
            raise ValueError(msg)
        BUILDERS[name] = fn
        return fn

    return deco


def build(name: str, params: dict[str, Any] | None = None) -> Classifier:
    """A fresh estimator by name. `params` override the builder's defaults."""
    from s6e7 import models  # noqa: F401  — importing the package registers every builder

    if name not in BUILDERS:
        msg = f"unknown model {name!r}; registered: {sorted(BUILDERS)}"
        raise KeyError(msg)
    return BUILDERS[name](dict(params or {}))
