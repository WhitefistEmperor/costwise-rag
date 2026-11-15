"""Prints the router section of results/offline.json (run the offline eval first)."""
import json
from pathlib import Path

r = json.loads((Path(__file__).resolve().parents[1] / "results" / "offline.json").read_text(encoding="utf-8"))
router = r["router"]
print(router["note"])
for name, v in router["routers"].items():
    t = v["test"]
    print(f"\n== {name}: threshold {v['threshold_from_tune']} (from tune)")
    print(f"  tune: accuracy {v['tune']['accuracy']} ({v['tune']['hard_to_large']}/{v['tune']['hard']} hard to large, "
          f"{v['tune']['easy_to_small']}/{v['tune']['easy']} easy kept small)")
    print(f"  test: accuracy {t['accuracy']} ({t['hard_to_large']}/{t['hard']} hard to large, "
          f"{t['easy_to_small']}/{t['easy']} easy kept small, {t['sent_to_large']}/{t['queries']} sent to large)")
    for c in v["test_by_threshold"]:
        print(f"    t={c['threshold']}: hard {c['hard_to_large']}/{c['hard']}, easy small {c['easy_to_small']}/{c['easy']}, "
              f"to large {c['sent_to_large']}/{c['queries']}")
