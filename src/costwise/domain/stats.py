"""Small statistics helpers shared by the request log and the live eval."""
from __future__ import annotations

import math
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the value at index ceil(p * n) - 1 of the sorted values, so for
    1..100 p50 is 50 and p95 is 95. `p` is in (0, 1]. None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    rank = math.ceil(round(p * n, 9))            # round first: 0.7 * 10 is 7.000000000000001
    return ordered[min(n - 1, max(0, rank - 1))]
