"""Price table and cost maths. Prices are USD per million tokens (input, output).

Providers that report the real cost per call (OpenRouter does) override this table; it is used
for the mock model, for the "what would the large model have cost" baseline, and as a fallback.
"""
from __future__ import annotations

PRICES_PER_M: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "anthropic/claude-opus-5.5": (5.00, 25.00),
    "mock/small": (0.10, 0.50),
    "mock/large": (2.00, 10.00),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int,
             prices: dict[str, tuple[float, float]] | None = None) -> float:
    """Cost of one call from the table. Unknown models cost 0 and should be added to the table."""
    inp, out = (prices or PRICES_PER_M).get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def has_price(model: str, prices: dict[str, tuple[float, float]] | None = None) -> bool:
    """True if the table (or `prices`) has a nonzero price for `model`. A paid run must refuse
    unpriced models, because their cost estimate is 0 and the budget cap would be blind."""
    inp, out = (prices or PRICES_PER_M).get(model, (0.0, 0.0))
    return inp > 0 or out > 0


def estimate_tokens(text: str) -> int:
    """Rough token count for the mock model and for pre-call budget checks: ~4 characters per token."""
    return max(1, len(text) // 4)
