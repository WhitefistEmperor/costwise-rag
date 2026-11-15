"""Prints a compact view of the cache section of the offline eval."""
from costwise.eval.offline import run

results, tuned = run()
cache = results["cache"]
print("tuned:", tuned)
print(f"stream: repeat rate {cache['repeat_rate']}, {cache['stream_seeds']} seeds")
for name, v in cache["variants"].items():
    print(f"\n== {name}: threshold {v['threshold_from_tune']} (from tune split)")
    print("  tune pairs :", v["tune_pairs"])
    print("  test pairs :", v["test_pairs"])
    print("  test stream:", v["test_stream"])
    for t, s in v["test_stream_by_threshold"].items():
        print(f"    t={t}: hits {s['hits']}, wrong {s['wrong_hits']}, repeat recall {s['repeat_recall']}")
