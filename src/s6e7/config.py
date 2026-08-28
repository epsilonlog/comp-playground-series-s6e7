"""Global constants. Frozen dataclasses for per-experiment config live alongside.

`SEED` is seeded into numpy, `random`, and every model. One value, one place — a
competition where two components disagree about the seed is a competition where no
comparison means anything.
"""

from __future__ import annotations

from typing import Final

SEED: Final[int] = 42

#: Physical cores are 4, threads 8. Six workers keeps the machine usable while running.
N_JOBS: Final[int] = 6
