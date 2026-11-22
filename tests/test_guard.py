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
    assert g.key_terms("3,000 and 1.30 and 800GB") == {"num:3000", "num:1.3", "num:800", "unit:gb"}
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


@pytest.mark.parametrize("a,b", [
    ("cost to migrate 800GB", "cost to migrate 800MB"),
    ("restore takes 6h", "restore takes 6 days"),
    ("links expire after 30 days", "links expire after 30 minutes"),
    ("upload limit of 2 TB", "upload limit of 2 gigabytes"),
    ("90-day retention", "90-hour retention"),
])
def test_guard_rejects_a_unit_only_change(a, b):
    assert not KeyTermGuard(set()).same_question(a, b)


@pytest.mark.parametrize("a,b", [
    ("cost to migrate 800GB", "cost to migrate 800 GB"),
    ("restore takes 6h", "restore takes 6 hours"),
    ("Team needs 90-day retention", "we need backups kept 90 days"),   # hyphen form counts as the unit
    ("cost to migrate 800GB", "what does moving 800 of storage cost"),   # unit dropped: an addition
    ("price of 2FA on Team", "team 2fa price"),
    ("does the v2 API allow it", "is it allowed in the v2 API"),
    ("billed monthly", "billed per month"),                             # no number: no unit term
])
def test_guard_units_follow_the_substitution_rule(a, b):
    assert KeyTermGuard(set()).same_question(a, b)


def test_cap_ratio_keeps_mostly_capitalised_words_and_drops_i():
    from costwise.adapters.verifiers import extract_entity_terms
    from costwise.domain.models import Chunk

    text = "t\nWhat Is the best way to learn Python?\nhow is python used. Is Python hard for I and me?\nwhere is Paris in winter"
    chunks = [Chunk("c", "d", "", text)]
    closed = extract_entity_terms(chunks)                      # the service's rule: any mid-sentence capital
    assert {"is", "python", "i"} <= closed
    open_rule = extract_entity_terms(chunks, 0.6)
    assert "python" in open_rule                               # capitalised in 2 of 3 mid-sentence uses
    assert "is" not in open_rule and "i" not in open_rule      # "is": capitalised 1 of 3 times
    assert "paris" in open_rule                                # capitalised every time it appears mid-sentence
