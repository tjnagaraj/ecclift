"""
Small-order completeness check for EccLift = ET(L(G)).

Claim checked (paper, Sec. 3, paragraph after the conjecture; also Abstract,
Contributions, and Conclusion): EccLift is a COMPLETE invariant on all 995
connected graphs of order <= 7, i.e. no two non-isomorphic such graphs share an
EccLift value.

Enumeration: networkx's graph atlas (networkx.generators.atlas.graph_atlas_g),
which lists all 1253 graphs on 0..7 nodes, exactly one representative per
isomorphism class (from Read & Wilson, "An Atlas of Graphs", 1998). We keep the
connected graphs of order 2..7. Since atlas entries are pairwise non-isomorphic,
two entries with equal EccLift profiles are a collision (EccLift fails to be
complete there).

EccLift is computed with ecclift_profile from eccentric_lift.py. Its entries are
ecc*trans of BFS hop distances, hence integers; they are stored as exact integer
tuples for bucketing.

Collisions are counted globally (across all orders, since EccLift equality does
not by itself force equal order) and within each order.

Writes results/results_small_order_completeness.{json,txt}.
"""

import json
import os
import sys
from collections import defaultdict

import networkx as nx
from networkx.generators.atlas import graph_atlas_g

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eccentric_lift import ecclift_profile

OUT_DIR = os.path.join(os.path.dirname(HERE), "results")
MIN_ORDER, MAX_ORDER = 2, 7


def ecclift_key(G):
    p = ecclift_profile(G)
    return tuple(int(round(x)) for x in p)


def main():
    graphs = [G for G in graph_atlas_g()
              if MIN_ORDER <= G.number_of_nodes() <= MAX_ORDER
              and nx.is_connected(G)]

    buckets = defaultdict(list)          # EccLift key -> list of atlas indices
    order_of = []
    for i, G in enumerate(graphs):
        order_of.append(G.number_of_nodes())
        buckets[ecclift_key(G)].append(i)

    # a collision = an unordered pair of distinct (non-isomorphic) atlas graphs
    # with equal EccLift
    def n_pairs(k):
        return k * (k - 1) // 2

    global_collision_pairs = sum(n_pairs(len(m)) for m in buckets.values())
    global_collision_buckets = sum(1 for m in buckets.values() if len(m) > 1)

    per_order = {}
    for n in range(MIN_ORDER, MAX_ORDER + 1):
        idx = [i for i, o in enumerate(order_of) if o == n]
        ob = defaultdict(int)
        for key, members in buckets.items():
            c = sum(1 for i in members if order_of[i] == n)
            if c:
                ob[key] = c
        per_order[str(n)] = {
            "connected_graphs": len(idx),
            "distinct_ecclift_values": len(ob),
            "collision_pairs_within_order": sum(n_pairs(c) for c in ob.values()),
        }

    examples = []
    for key, members in buckets.items():
        if len(members) > 1 and len(examples) < 10:
            examples.append([nx.to_graph6_bytes(graphs[i], header=False)
                             .decode().strip() for i in members])

    complete = global_collision_pairs == 0
    R = {
        "purpose": "EccLift completeness on all connected graphs of order 2..7",
        "enumeration": "networkx graph_atlas_g (one graph per isomorphism class, "
                       "0..7 nodes), filtered to connected graphs of order 2..7",
        "n_graphs": len(graphs),
        "n_distinct_ecclift_values": len(buckets),
        "collision_buckets": global_collision_buckets,
        "collision_pairs": global_collision_pairs,
        "ecclift_complete": complete,
        "per_order": per_order,
        "collision_examples_graph6": examples,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results_small_order_completeness.json"), "w") as f:
        json.dump(R, f, indent=2)

    lines = ["EccLift completeness on connected graphs of order 2..7 "
             "(networkx graph atlas)",
             "order | connected graphs | distinct EccLift values | collision pairs"]
    for n, d in per_order.items():
        lines.append(f"{n:>5} | {d['connected_graphs']:>16} | "
                     f"{d['distinct_ecclift_values']:>23} | "
                     f"{d['collision_pairs_within_order']:>15}")
    lines.append(f"TOTAL: {len(graphs)} graphs, {len(buckets)} distinct EccLift "
                 f"values, {global_collision_pairs} collision pairs (all orders "
                 f"pooled) -> complete: {complete}")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "results_small_order_completeness.txt"), "w") as f:
        f.write(txt)
    print(txt, end="")
    print("wrote results/results_small_order_completeness.{json,txt}")


if __name__ == "__main__":
    main()
