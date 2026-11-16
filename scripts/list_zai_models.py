"""List available models from Z.ai (or the China platform via COSTWISE_ZAI_BASE_URL).

    python scripts/list_zai_models.py

Reads ZAI_API_KEY from the environment. Prints model ids, filtered to show Flash / free models
first. The model id is what you pass to COSTWISE_SMALL_MODEL and COSTWISE_LARGE_MODEL.
"""
from __future__ import annotations

import os
import sys

import httpx

BASE_URL = os.environ.get("COSTWISE_ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
KEY = os.environ.get("ZAI_API_KEY", "")

if not KEY:
    sys.exit("ZAI_API_KEY is not set in the environment")

try:
    r = httpx.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=15)
    r.raise_for_status()
    data = r.json()
except Exception as e:
    sys.exit(f"request failed: {e}")

models = data.get("data") or []
if not models:
    print("No models found in response:", data)
    sys.exit(1)

ids = sorted(m.get("id", "") for m in models if isinstance(m, dict))

print(f"Found {len(ids)} model(s) at {BASE_URL}:\n")
flash = [i for i in ids if "flash" in i.lower() or "free" in i.lower()]
rest  = [i for i in ids if i not in flash]

if flash:
    print("  -- Flash / free models (use these) --")
    for i in flash:
        print(f"  {i}")
    print()
if rest:
    print("  -- Other models --")
    for i in rest:
        print(f"  {i}")
