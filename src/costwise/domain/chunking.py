"""Splits markdown documents into one chunk per `##` section, prefixed with the doc title."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .models import Chunk


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def chunk_markdown(doc: str, text: str, doc_hash: str) -> list[Chunk]:
    title, heading = doc, ""
    sections: list[str] = []
    body: list[str] = []

    def flush() -> None:
        content = " ".join(line.strip() for line in body if line.strip())
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


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        data = path.read_bytes()
        chunks.extend(chunk_markdown(path.name, data.decode("utf-8"), file_hash(data)))
    if not chunks:
        raise ValueError(f"no markdown documents found in {corpus_dir}")
    return chunks


def doc_hashes(chunks: list[Chunk]) -> dict[str, str]:
    return {c.doc: c.doc_hash for c in chunks}
