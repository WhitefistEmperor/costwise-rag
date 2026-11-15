"""Reads the markdown corpus from disk. The chunking itself is pure and lives in domain/."""
from __future__ import annotations

from pathlib import Path

from ..domain.chunking import chunk_markdown, file_hash
from ..domain.models import Chunk


def load_corpus(corpus_dir: Path | str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        data = path.read_bytes()
        chunks.extend(chunk_markdown(path.name, data.decode("utf-8"), file_hash(data)))
    if not chunks:
        raise ValueError(f"no markdown documents found in {corpus_dir}")
    return chunks
