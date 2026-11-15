"""Splits markdown documents into one chunk per `##` section, prefixed with the doc title.
Pure: reading the files is done by adapters/corpus.py."""
from __future__ import annotations

import hashlib

from .models import Chunk


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def chunk_markdown(doc: str, text: str, doc_hash: str) -> list[Chunk]:
    title, heading = doc, ""
    sections: list[str] = []
    body: list[str] = []

    def flush() -> None:
        # Line breaks are kept: embedders and BM25 treat them as spaces, and the guard's entity
        # extraction needs them to know where a line or bullet starts.
        content = "\n".join(line.strip() for line in body if line.strip())
        if content:
            head = f"{title} - {heading}" if heading else title
            sections.append(f"{head}\n{content}")

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    flush()
    return [Chunk(id=f"{doc}#{i}", doc=doc, doc_hash=doc_hash, text=s) for i, s in enumerate(sections)]


def doc_hashes(chunks: list[Chunk]) -> dict[str, str]:
    return {c.doc: c.doc_hash for c in chunks}
