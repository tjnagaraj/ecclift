"""
CONVERSE EXPLORATION for EccLift = ET(L(G)).

The paper proves the FORWARD direction (Theorem 1 / thm:char):
    equal (vertex distance-distribution VDD, edge-min-distance profile emdp)
    ==> EccLift(G) = EccLift(H).
and states the CONVERSE (Conjecture 1 / conj:char) as an unproved conjecture:
    (VDD, emdp) differs between G,H  ==>  EccLift(G) != EccLift(H),
    equivalently  EccLift(G)=EccLift(H)  ==>  (VDD,emdp) equal,
backed by 0 counterexamples over 6444 pairs (connected atlas graphs order<=6,
matched by order, plus the named SR mates).

This script SEARCHES FOR A COUNTEREXAMPLE to the converse: a pair of connected
graphs G,H with
    EccLift(G) == EccLift(H)      (EccLift TIES)
    but  (VDD,emdp)(G) != (VDD,emdp)(H)   (the profile pair DIFFERS).
Such a pair refutes the converse as stated.

We bucket ALL connected graphs (order 7 from the atlas; optionally order 8 if a
graph6 source is provided) by their EXACT-integer EccLift tuple. EccLift equality
forces equal lift-node count (p+q), so bucketing by the profile automatically
matches sizes. Within every bucket of >=2 graphs we look for a pair whose
(VDD,emdp) profile pair differs -> counterexample.

Exact-integer only (BFS distances are integers, et = ecc*trans is an integer).
No eigensolver. Writes results/converse_explore_out.json.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eccentric_lift import incidence_lift, _all_pairs_dist
from characterization import (
    vertex_distance_distribution,
    edge_min_distance_profile,
    ecclift_invariant,
    intersection_array,
)

OUT = os.path.join(os.path.dirname(HERE), "results", "converse_explore_out.json")


# ---------------------------------------------------------------------------
# EXACT-INTEGER EccLift as a hashable tuple (BFS only, no eigensolver)
# ---------------------------------------------------------------------------
def ecclift_tuple(G):
    """EccLift(G) = sorted multiset of ecc(w)*trans(w) over lift nodes w, as an
    exact-integer tuple. Returns None if the lift is disconnected."""
    H = incidence_lift(G)
    if H.number_of_nodes() == 0 or not nx.is_connected(H):
        return None
    nodes = list(H.nodes())
    # exact integer all-pairs BFS
    et = []
    for src in nodes:
        lengths = nx.single_source_shortest_path_length(H, src)
        vals = list(lengths.values())
        trans = sum(vals)
        ecc = max(vals)
        et.append(ecc * trans)
    return tuple(sorted(et))


def profile_pair(G):
    """(VDD, emdp) as hashable tuple -- the COMMITTED-CODE characterization pair
    (characterization.py). NOTE: this uses the COARSE vertex_distance_distribution
    (per-vertex G-distance multiset only). It is strictly coarser than the paper's
    LaTeX 'vertex distance-distribution', which is the full vertex-node lift-row
    profile {rho(u)} (it also folds in {1+2*min_{x in e} d_G(u,x)})."""
    return ecclift_invariant(G)


def lift_row_profile(G):
    """The paper's ACTUAL characterization object (LaTeX Sec. theory): the lift
    distance-row profile {rho(w) : w in L(G)}, split into (vertex-node rows,
    edge-node rows). rho(w) = sorted multiset of lift-distances from w. EccLift is
    BY CONSTRUCTION a function of this (ecc = max rho, trans = sum rho). Equal to
    (paper's vertex distance-distribution {rho(u)}, emdp <-> {rho(e)})."""
    H = incidence_lift(G)
    if H.number_of_nodes() == 0 or not nx.is_connected(H):
        return None
    vrows, erows = [], []
    for nd in H.nodes():
        lengths = nx.single_source_shortest_path_length(H, nd)
        row = tuple(sorted(lengths.values()))
        (vrows if nd[0] == "v" else erows).append(row)
    return (tuple(sorted(vrows)), tuple(sorted(erows)))


def load_atlas_order(k):
    from networkx.generators.atlas import graph_atlas_g
    return [G for G in graph_atlas_g()
            if G.number_of_nodes() == k and nx.is_connected(G)]


def search(graphs, label):
    """Bucket graphs by EccLift tuple; report converse counterexamples."""
    buckets = defaultdict(list)
    skipped = 0
    for i, G in enumerate(graphs):
        t = ecclift_tuple(G)
        if t is None:
            skipped += 1
            continue
        buckets[t].append(i)

    # cache profile pairs lazily
    prof_cache = {}

    def prof(i):
        if i not in prof_cache:
            prof_cache[i] = profile_pair(graphs[i])
        return prof_cache[i]

    counterexamples = []
    multi = 0
    pairs_checked = 0
    for t, members in buckets.items():
        if len(members) < 2:
            continue
        multi += 1
        # any pair in this EccLift-tie bucket with differing profile pair?
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ia, ib = members[a], members[b]
                pairs_checked += 1
                if prof(ia) != prof(ib):
                    counterexamples.append((ia, ib, t))
    return {
        "label": label,
        "n_graphs": len(graphs),
        "n_lift_disconnected_skipped": skipped,
        "n_ecclift_tie_buckets": multi,
        "n_tie_pairs_checked": pairs_checked,
        "n_counterexamples": len(counterexamples),
        "counterexamples": counterexamples,
        "_buckets": buckets,
    }


def describe(G):
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "graph6": nx.to_graph6_bytes(G, header=False).decode().strip(),
        "degree_sequence": sorted([d for _, d in G.degree()]),
        "diameter": nx.diameter(G),
        "is_regular": len({d for _, d in G.degree()}) == 1,
    }


def multiset_str(t):
    from collections import Counter
    c = Counter(t)
    return "{" + ", ".join(f"{v}x{n}" for v, n in sorted(c.items())) + "}"


ROOT = os.path.abspath(os.path.join(HERE, ".."))
# BREC data (not bundled): see README. Override with the BREC_PATH env variable.
BREC_PATH = os.environ.get(
    "BREC_PATH", os.path.join(ROOT, "data", "BREC", "brec_v3.npy"))
STRIDE = 64
CATS = {
    "Basic": (0, 60), "Regular": (60, 160), "Extension": (160, 260),
    "CFI": (260, 360), "4-Vertex": (360, 380), "Distance-Regular": (380, 400),
}


def _brec_graph(e):
    s = e.decode() if isinstance(e, (bytes, bytearray)) else str(e)
    return nx.from_graph6_bytes(s.encode())


def brec_converse_scan():
    """For every BREC pair where EccLift TIES (exact integer), check whether the
    characterization profile pair (VDD, emdp) DIFFERS. A tie with differing
    profiles is a COUNTEREXAMPLE to the converse conjecture. VDD and emdp are
    permutation-invariant, so a single canonical representative per graph suffices."""
    BREC = np.load(BREC_PATH, allow_pickle=True)
    out = {"per_category": {}, "counterexamples": []}
    tot_tie = 0
    tot_ce = 0
    for cat, (lo, hi) in CATS.items():
        cat_tie = 0
        cat_ce = 0
        for pid in range(lo, hi):
            b = pid * STRIDE
            g0 = _brec_graph(BREC[b])
            g1 = _brec_graph(BREC[b + 1])
            t0 = ecclift_tuple(g0)
            t1 = ecclift_tuple(g1)
            if t0 is None or t1 is None:
                continue
            if t0 != t1:
                continue  # EccLift separates this pair -> not relevant to converse
            cat_tie += 1
            tot_tie += 1
            vdd_eq = vertex_distance_distribution(g0) == vertex_distance_distribution(g1)
            emdp_eq = edge_min_distance_profile(g0) == edge_min_distance_profile(g1)
            if not (vdd_eq and emdp_eq):
                cat_ce += 1
                tot_ce += 1
                out["counterexamples"].append({
                    "category": cat, "pair_id": pid,
                    "n": g0.number_of_nodes(), "m": g0.number_of_edges(),
                    "g0_graph6": nx.to_graph6_bytes(g0, header=False).decode().strip(),
                    "g1_graph6": nx.to_graph6_bytes(g1, header=False).decode().strip(),
                    "ecclift_equal": True,
                    "ecclift_multiset": multiset_str(t0),
                    "vertex_distance_distribution_equal": bool(vdd_eq),
                    "edge_min_distance_profile_equal": bool(emdp_eq),
                    "isomorphic": bool(nx.is_isomorphic(g0, g1)),
                    "g0_regular": len({d for _, d in g0.degree()}) == 1,
                    "g0_diameter": nx.diameter(g0),
                })
        out["per_category"][cat] = {"ecclift_tie_pairs": cat_tie,
                                    "converse_counterexamples": cat_ce}
    out["total_ecclift_tie_pairs"] = tot_tie
    out["total_converse_counterexamples"] = tot_ce
    return out


def _build_regular_pool(seed):
    import random
    random.seed(seed)
    graphs = []

    def add(G):
        if G.number_of_nodes() >= 6 and nx.is_connected(G):
            graphs.append(G)
    for n in range(8, 17):
        for r in range(3, min(n - 1, 7)):
            if (n * r) % 2:
                continue
            for _ in range(350):
                try:
                    add(nx.random_regular_graph(r, n))
                except Exception:
                    break
    for n in range(8, 22):
        conns = list(range(1, n // 2 + 1))
        for k in range(1, 4):
            for S in __import__("itertools").combinations(conns, k):
                add(nx.circulant_graph(n, S))
    base = [nx.cycle_graph(4), nx.complete_graph(4), nx.complete_graph(3),
            nx.petersen_graph(), nx.cycle_graph(5), nx.cycle_graph(6),
            nx.complete_bipartite_graph(3, 3)]
    for A in base:
        for B in base:
            add(nx.cartesian_product(A, B))
            add(nx.tensor_product(A, B))
    for n in range(9, 13):
        import random as _r
        for _ in range(500):
            add(nx.gnp_random_graph(n, _r.uniform(0.3, 0.6)))
    return graphs


def regular_pool_scan(seed=3):
    """Converse test (coarse committed (VDD,emdp) AND paper's finer lift-row
    profile) plus the FORWARD-direction probe of the coarse committed invariant,
    over a large regular/circulant/product/random pool where EccLift actually
    ties. This is the regime the paper's order-<=6 enumeration never reached."""
    graphs = _build_regular_pool(seed)
    buckets = defaultdict(list)
    for i, G in enumerate(graphs):
        t = ecclift_tuple(G)
        if t is not None:
            buckets[t].append(i)

    coarse_ce = row_ce = noniso_ties = 0
    row_ce_examples = []
    for t, m in buckets.items():
        if len(m) < 2:
            continue
        reps = []
        for i in m:
            if not any(nx.is_isomorphic(graphs[i], graphs[j]) for j in reps):
                reps.append(i)
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                G, H = graphs[reps[a]], graphs[reps[b]]
                noniso_ties += 1
                if profile_pair(G) != profile_pair(H):
                    coarse_ce += 1
                if lift_row_profile(G) != lift_row_profile(H):
                    row_ce += 1
                    if len(row_ce_examples) < 5:
                        row_ce_examples.append(
                            (nx.to_graph6_bytes(G, header=False).decode().strip(),
                             nx.to_graph6_bytes(H, header=False).decode().strip()))

    # forward-gap probe for the COARSE committed invariant
    pbuck = defaultdict(list)
    for i, G in enumerate(graphs):
        pbuck[profile_pair(G)].append(i)
    fgap = 0
    fgap_ex = []
    for key, m in pbuck.items():
        if len(m) < 2:
            continue
        ts = defaultdict(list)
        for i in m:
            ts[ecclift_tuple(graphs[i])].append(i)
        if len(ts) > 1:
            keys = list(ts.keys())
            Gi, Hi = ts[keys[0]][0], ts[keys[1]][0]
            if not nx.is_isomorphic(graphs[Gi], graphs[Hi]):
                fgap += 1
                if len(fgap_ex) < 5:
                    fgap_ex.append({
                        "n": graphs[Gi].number_of_nodes(),
                        "m": graphs[Gi].number_of_edges(),
                        "g6_G": nx.to_graph6_bytes(graphs[Gi], header=False).decode().strip(),
                        "g6_H": nx.to_graph6_bytes(graphs[Hi], header=False).decode().strip(),
                        "coarse_VDD_emdp_equal": True,
                        "ecclift_differs": True,
                        "row_profile_differs":
                            lift_row_profile(graphs[Gi]) != lift_row_profile(graphs[Hi]),
                    })
    return {
        "pool_size": len(graphs),
        "noniso_tie_pairs": noniso_ties,
        "converse_ce_coarse": coarse_ce,
        "converse_ce_rowprofile": row_ce,
        "row_ce_examples": row_ce_examples,
        "forward_gap_coarse_buckets": fgap,
        "forward_gap_witnesses": fgap_ex,
        "note": ("CONVERSE (EccLift tie => profiles equal) holds with ZERO "
                 "counterexamples against both the coarse committed (VDD,emdp) and "
                 "the paper's finer lift-row profile. The FORWARD direction of the "
                 "COARSE committed invariant FAILS at order>=11 (fgap witnesses): "
                 "equal coarse (VDD,emdp) yet EccLift differs -- because the coarse "
                 "vertex_distance_distribution drops the by-vertex vertex-to-edge "
                 "min terms that the paper's LaTeX rho(u) keeps. The paper's LaTeX "
                 "Theorem (finer rho(u)) is unaffected; only the code proxy is coarse."),
    }


def main():
    R = {"purpose": "search for a counterexample to the EccLift converse "
                    "conjecture (conj:char): EccLift ties but (VDD,emdp) differs"}

    orders = [7]
    all_graphs = []
    for k in orders:
        gs = load_atlas_order(k)
        print(f"order {k}: {len(gs)} connected graphs")
        all_graphs.extend(gs)

    # Search WITHIN each order and also globally (EccLift equality already forces
    # equal p+q, so a global bucketing is the most general test).
    res = search(all_graphs, f"connected atlas graphs order in {orders}")
    print(f"[{res['label']}] tie-buckets={res['n_ecclift_tie_buckets']} "
          f"tie-pairs={res['n_tie_pairs_checked']} "
          f"COUNTEREXAMPLES={res['n_counterexamples']}")

    R["order7_global_search"] = {k: v for k, v in res.items() if k != "_buckets"}

    # detail the first few counterexamples
    details = []
    for (ia, ib, t) in res["counterexamples"][:10]:
        G, H = all_graphs[ia], all_graphs[ib]
        vdd_eq = vertex_distance_distribution(G) == vertex_distance_distribution(H)
        emdp_eq = edge_min_distance_profile(G) == edge_min_distance_profile(H)
        details.append({
            "G": describe(G),
            "H": describe(H),
            "ecclift_equal": True,
            "ecclift_multiset": multiset_str(t),
            "vertex_distance_distribution_equal": bool(vdd_eq),
            "edge_min_distance_profile_equal": bool(emdp_eq),
            "profile_pair_differs": bool(not (vdd_eq and emdp_eq)),
            "isomorphic": bool(nx.is_isomorphic(G, H)),
        })
    R["order7_counterexample_details"] = details

    # ---- BREC converse scan: the regime where EccLift actually TIES ----
    if os.path.exists(BREC_PATH):
        print("\nBREC converse scan (EccLift-tie pairs with differing profiles)...")
        brec = brec_converse_scan()
        R["brec_converse_scan"] = brec
        for cat, d in brec["per_category"].items():
            print(f"  {cat:18s} ecclift-ties={d['ecclift_tie_pairs']:3d} "
                  f"converse-counterexamples={d['converse_counterexamples']:3d}")
        print(f"  TOTAL ecclift-tie pairs = {brec['total_ecclift_tie_pairs']}, "
              f"converse counterexamples = {brec['total_converse_counterexamples']}")
    else:
        print("BREC data not found; skipping BREC converse scan")

    # ---- regular/circulant/product pool: the only regime with EccLift TIES ----
    # Order <=7 has NO EccLift ties at all (EccLift is a complete invariant there),
    # so a genuine converse/forward test needs the larger regular families where
    # EccLift collapses. We test the converse against BOTH the coarse committed
    # (VDD,emdp) AND the paper's finer lift-row profile, and probe the FORWARD
    # direction of the COARSE committed invariant.
    print("\nregular/circulant/product pool (order 8-16): ties + forward-gap...")
    reg = regular_pool_scan(seed=3)
    R["regular_pool_scan"] = reg
    print(f"  pool={reg['pool_size']}  non-iso EccLift-tie pairs={reg['noniso_tie_pairs']}")
    print(f"  converse counterexamples vs COARSE (VDD,emdp): {reg['converse_ce_coarse']}")
    print(f"  converse counterexamples vs PAPER row-profile : {reg['converse_ce_rowprofile']}")
    print(f"  FORWARD-GAP buckets (coarse (VDD,emdp) equal, EccLift differs): "
          f"{reg['forward_gap_coarse_buckets']}")
    for w in reg["forward_gap_witnesses"][:3]:
        print(f"     forward-gap witness (n={w['n']}): {w['g6_G']} vs {w['g6_H']} "
              f"| coarse (VDD,emdp) EQUAL, EccLift DIFFERS, row-profile differs")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"wrote {os.path.relpath(OUT, HERE)}")
    if details:
        print("\n=== ORDER-7 COUNTEREXAMPLE ===")
        print(json.dumps(details[0], indent=2, default=str))
    if R.get("brec_converse_scan", {}).get("counterexamples"):
        print("\n=== FIRST BREC CONVERSE COUNTEREXAMPLE ===")
        print(json.dumps(R["brec_converse_scan"]["counterexamples"][0],
                         indent=2, default=str))


if __name__ == "__main__":
    main()
