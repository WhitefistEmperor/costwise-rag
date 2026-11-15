"""KeyTermGuard: numbers glued to letters, exact number comparison, and entity terms that do
not pick up the first word of a line or bullet."""
from __future__ import annotations

import pytest

from costwise.adapters.verifiers import KeyTermGuard, extract_entity_terms
from costwise.domain.chunking import chunk_markdown


@pytest.mark.parametrize("a,b", [
    ("cost to migrate 800GB", "cost to migrate 500GB"),
    ("restore takes 6h", "restore takes 4h"),
    ("does the v2 API allow it", "does the v3 API allow it"),
    ("links expire after 30d", "links expire after 14d"),
    ("Business allows 100000000.5 requests", "Business allows 100000000.6 requests"),   # beyond 6 digits
    ("rate limit of 3000001", "rate limit of 3000002"),
])
def test_guard_rejects_different_numbers(a, b):
    assert not KeyTermGuard(set()).same_question(a, b)


@pytest.mark.parametrize("a,b", [
    ("rate limit of 3,000 requests", "rate limit of 3000 requests"),
    ("1.30 TB on Team", "1.3 TB on Team"),
    ("migrate 800GB", "migrate 800 GB"),
])
def test_guard_normalises_equal_numbers(a, b):
    assert KeyTermGuard(set()).same_question(a, b)


def test_guard_number_terms_are_exact():
    g = KeyTermGuard(set())
    assert g.key_terms("3,000 and 1.30 and 800GB") == {"num:3000", "num:1.3", "num:800"}
    assert g.key_terms("1,234,567.50") == {"num:1234567.5"}


DOC = """# Acme Help

## Navigation
- Settings holds the account options.
- Billing shows invoices from Stripe.
* Export works for every plan.
1. Restore jobs run on the Recovery Center page.
Plain sentence about Frankfurt hosting.
Developers can call the API.
"""


def test_line_and_bullet_initial_words_are_not_entity_terms():
    terms = extract_entity_terms(chunk_markdown("help.md", DOC, "h"))
    for word in ("settings", "billing", "export", "restore", "plain", "developers"):
        assert word not in terms
    for word in ("stripe", "center", "frankfurt", "api"):    # genuinely capitalised mid-sentence
        assert word in terms


def test_chunk_text_keeps_line_breaks_inside_a_section():
    chunk = chunk_markdown("help.md", DOC, "h")[0]
    assert chunk.text.splitlines()[1] == "- Settings holds the account options."
