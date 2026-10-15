"""Runtime/memory scaling of EccLift (paper Appendix A, Table "EccLift runtime/memory
scaling"): wall-clock runtime and peak memory of EccLift vs. graph size, measuring the
O(|E|^2) analysis empirically, using the implementation (ecclift_profile from
eccentric_lift.py) unchanged.

Families: random r-regular graphs, r in {3, 8}, n from 100 to 3200 (sparse regime,
|E| = rn/2) -- the regime the invariant targets. Median wall time of 3 runs;
peak Python-heap memory via tracemalloc. Also reports the empirical scaling exponent
between successive sizes (log time-ratio / log |E|-ratio), to compare against the
O(|E|^2) bound (exponent 2 in |E| at fixed density <=> slope ~2 in n).

Writes results/results_runtime_scaling.{txt,json}. CPU only.
"""
import json
import os
import statistics
import sys
import time
import tracemalloc

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eccentric_lift import ecclift_profile

OUT = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(OUT, exist_ok=True)
SIZES = [100, 200, 400, 800, 1600, 3200]
DEGREES = [3, 8]
SEED = 7


def one_graph(r, n):
    return nx.random_regular_graph(r, n, seed=SEED)


def measure(G):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ecclift_profile(G)
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    ecclift_profile(G)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(times), peak


rows = []
for r in DEGREES:
    prev = None
    for n in SIZES:
        if r >= n:
            continue
        G = one_graph(r, n)
        m = G.number_of_edges()
        t, peak = measure(G)
        slope = None
        if prev is not None:
            import math
            slope = math.log(t / prev[1]) / math.log(m / prev[0])
        rows.append({"degree": r, "n": n, "edges": m,
                     "median_sec": round(t, 4), "peak_mem_mb": round(peak / 1e6, 2),
                     "empirical_exponent_vs_prev": round(slope, 2) if slope else None})
        print(f"r={r} n={n:>5} |E|={m:>6}  t={t:8.3f}s  peak={peak/1e6:8.2f}MB"
              f"  slope={slope and f'{slope:.2f}' or '--'}", flush=True)
        prev = (m, t)

json.dump({"seed": SEED, "runs_per_point": 3, "rows": rows},
          open(os.path.join(OUT, "results_runtime_scaling.json"), "w"), indent=2)
lines = ["EccLift runtime/memory scaling (committed ecclift_profile; random r-regular graphs, seed 7)",
         "degree |     n |    |E| | median time (s) | peak mem (MB) | empirical exponent vs prev size"]
for row in rows:
    lines.append(f"{row['degree']:>6} | {row['n']:>5} | {row['edges']:>6} | "
                 f"{row['median_sec']:>15} | {row['peak_mem_mb']:>13} | "
                 f"{row['empirical_exponent_vs_prev'] or '--'}")
lines.append("O(|E|^2) predicts exponent ~2.0 between successive sizes at fixed degree.")
open(os.path.join(OUT, "results_runtime_scaling.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/results_runtime_scaling.{txt,json}")
