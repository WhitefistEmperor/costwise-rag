"""Runtime settings. Every value can be overridden with a COSTWISE_* environment variable."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"COSTWISE_{name}", default)


@dataclass
class Settings:
    corpus_dir: Path = field(default_factory=lambda: Path(_env("CORPUS_DIR", str(ROOT / "data" / "corpus"))))
    questions_path: Path = ROOT / "data" / "questions.json"
    tuned_path: Path = ROOT / "data" / "tuned.json"
    state_dir: Path = field(default_factory=lambda: Path(_env("STATE_DIR", str(ROOT / ".state"))))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    small_model: str = field(default_factory=lambda: _env("SMALL_MODEL", "anthropic/claude-haiku-5.5"))
    large_model: str = field(default_factory=lambda: _env("LARGE_MODEL", "anthropic/claude-sonnet-5.5"))
    # "mock" (default, free, offline) or "openrouter" (paid; needs OPENROUTER_API_KEY in the environment).
    llm_provider: str = field(default_factory=lambda: _env("LLM", "mock"))
    openrouter_base_url: str = field(default_factory=lambda: _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    # "zai" provider: free GLM Flash on Z.ai. Base URL for Chinese platform: https://open.bigmodel.cn/api/paas/v4
    bharatcode_base_url: str = field(default_factory=lambda: _env("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1"))
    zai_base_url: str = field(default_factory=lambda: _env("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"))
    router: str = field(default_factory=lambda: _env("ROUTER", "learned"))     # learned | heuristic | none
    default_route: str = field(default_factory=lambda: _env("DEFAULT_ROUTE", "large"))   # when router is none
    cache_enabled: bool = field(default_factory=lambda: _env("CACHE", "1") not in ("0", "false", "no"))
    top_k: int = 4
    cache_threshold: float = 0.90          # accept threshold; the cheap verifier alone decides above it
    cache_candidate_threshold: float = 0.90   # band [candidate, accept) needs the strong verifier too
    cache_strong_verifier: str = field(default_factory=lambda: _env("STRONG_VERIFIER", "none"))  # none | llm
    cache_ttl_s: float = 24 * 3600
    route_threshold: float = 0.5
    max_answer_tokens: int = 300

    @property
    def cache_db(self) -> Path:
        return self.state_dir / "cache.db"

    @property
    def request_log_db(self) -> Path:
        return self.state_dir / "requests.db"

    def load_tuned(self) -> dict:
        """Values fitted by the offline eval (cache threshold, router weights), if present."""
        if self.tuned_path.exists():
            return json.loads(self.tuned_path.read_text(encoding="utf-8"))
        return {}
