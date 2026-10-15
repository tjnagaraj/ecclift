"""
Full BREC separation audit for EccLift (paper Section 5, Table "BREC separation audit").

Applies the eccentric-transmission-of-the-incidence-lift PE (EccLift) across the
whole BREC benchmark (data/BREC/brec_v3.npy, 800 pairs x 32 stored
relabellings x 2 graphs = 51200 graph6 strings), and COUNTS how many pairs
EccLift separates that plain-Laplacian AND RWSE tie on.

Leak guard: one invariant fingerprint is required across the stored
relabellings of each graph (K of them are sampled); a graph whose fingerprint
varies under relabelling is a LEAK -> not counted as a separation. BREC category
ranges are BREC's canonical layout.

All EccLift numbers are exact BFS distances on the incidence lift (no
eigensolver in EccLift). Plain-Laplacian and RWSE are the eigendecomposition /
random-walk baselines EccLift is compared against.

Scope: SEPARATION / expressiveness, NOT property-prediction accuracy. This
audit measures separation only.

Writes results/results_brec.json.
"""

import json
import os
import sys
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eccentric_lift import (
    incidence_lift,
    eccentric_transmission_profile,
    ecclift_profile,
)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
# BREC data (not bundled): see README. Override with the BREC_PATH env variable.
BREC_PATH = os.environ.get(
    "BREC_PATH", os.path.join(ROOT, "data", "BREC", "brec_v3.npy"))
OUT = os.path.join(ROOT, "results", "results_brec.json")

STRIDE = 64          # 32 relabellings x 2 graphs per pair
K = 4                # relabellings sampled per graph for the leak-guard
ROUND = 5

# BREC canonical category layout (pair indices).
CATS = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex": (360, 380),
    "Distance-Regular": (380, 400),
}


def G(e):
    s = e.decode() if isinstance(e, (bytes, bytearray)) else str(e)
    return nx.from_graph6_bytes(s.encode())


# ---------------- invariant fingerprints ----------------
def _lap_spectrum(g):
    A = nx.to_numpy_array(g, dtype=float)
    return np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A))


def lap_fp(g):
    return tuple(np.round(_lap_spectrum(g), ROUND))


def rwse_fp(g, k=8):
    A = nx.to_numpy_array(g, dtype=float)
    d = A.sum(1)
    d[d == 0] = 1.0
    P = A / d[:, None]
    out = []
    M = np.eye(A.shape[0])
    for _ in range(k):
        M = M @ P
        out.append(tuple(np.round(np.sort(np.diag(M)), ROUND)))
    return tuple(out)


def ecclift_fp(g):
    """EccLift fingerprint: ET( L(g) ) rounded, or None if the lift is
    disconnected (distances undefined -> cannot count as a separation)."""
    p = ecclift_profile(g)
    if p is None:
        return None
    return tuple(np.round(p, ROUND))


def et_of_G_fp(g):
    """Eccentric transmission on G itself (the collapse-on-regular baseline)."""
    p = eccentric_transmission_profile(g)
    if p is None:
        return None
    return tuple(np.round(p, ROUND))


# ---------------- leak-guarded separation over stored relabellings ----------------
def sep(fp, g0, g1):
    """Return True/False/None. None = one graph's fingerprint LEAKS (varies
    under the stored relabellings) or is undefined -> not a valid separation."""
    f0 = {fp(x) for x in g0}
    f1 = {fp(x) for x in g1}
    if None in f0 or None in f1:
        return None
    if len(f0) > 1 or len(f1) > 1:
        return None  # leak: invariant should be constant across relabellings
    return next(iter(f0)) != next(iter(f1))


def main():
    if not os.path.exists(BREC_PATH):
        sys.exit(f"BREC data not found at {BREC_PATH}; see README (Setup).")
    BREC = np.load(BREC_PATH, allow_pickle=True)
    n_pairs_total = len(BREC) // STRIDE
    print(f"BREC loaded: {len(BREC)} graph6 strings = {n_pairs_total} pairs "
          f"| leak-guard K={K} relabellings/graph")
    print()

    encoders = {
        "plain-Lap": lap_fp,
        "RWSE": rwse_fp,
        "ET(G)": et_of_G_fp,
        "EccLift": ecclift_fp,
    }

    # global counters
    tot = {k: 0 for k in encoders}     # #pairs each encoder separates (leak-safe)
    tot["n"] = 0
    tot["ecclift_leak"] = 0
    tot["ecclift_disc_lift"] = 0
    # THE headline counters
    both_tie = 0                        # plain-Lap AND RWSE both tie (valid, non-leak)
    ecclift_sep_where_both_tie = 0      # ... and EccLift separates
    ecclift_new_pairs = []             # per-pair record of the strict wins

    per_cat = {}

    for cat, (lo, hi) in CATS.items():
        c = {k: 0 for k in encoders}
        cat_both_tie = 0
        cat_new = 0
        for pid in range(lo, hi):
            b = pid * STRIDE
            g0 = [G(BREC[b + 2 * k]) for k in range(K)]
            g1 = [G(BREC[b + 2 * k + 1]) for k in range(K)]

            res = {k: sep(fp, g0, g1) for k, fp in encoders.items()}
            for k in encoders:
                if res[k] is True:
                    c[k] += 1
            if res["EccLift"] is None:
                tot["ecclift_leak"] += 1
                # distinguish disconnected-lift (shouldn't happen for connected G)
                if ecclift_fp(g0[0]) is None:
                    tot["ecclift_disc_lift"] += 1

            lap_tie = (res["plain-Lap"] is False)
            rwse_tie = (res["RWSE"] is False)
            if lap_tie and rwse_tie:
                cat_both_tie += 1
                both_tie += 1
                if res["EccLift"] is True:
                    ecclift_sep_where_both_tie += 1
                    cat_new += 1
                    ecclift_new_pairs.append({
                        "category": cat, "pair_id": pid,
                        "n0": g0[0].number_of_nodes(),
                    })
            tot["n"] += 1

        for k in encoders:
            tot[k] += c[k]
        per_cat[cat] = {
            "n_pairs": hi - lo,
            **{f"{k}_separates": c[k] for k in encoders},
            "both_Lap_and_RWSE_tie": cat_both_tie,
            "ecclift_separates_where_both_tie": cat_new,
        }
        print(f"{cat:18s} n={hi-lo:4d} | "
              + "  ".join(f"{k}={c[k]:3d}" for k in encoders)
              + f" || both-tie={cat_both_tie:3d}  EccLift-new={cat_new:3d}")

    print()
    print("=" * 72)
    print(f"TOTAL pairs                              : {tot['n']}")
    for k in encoders:
        print(f"  {k:12s} separates (leak-safe)      : {tot[k]}")
    print(f"  EccLift leak/undefined (not counted)   : {tot['ecclift_leak']} "
          f"(disc-lift: {tot['ecclift_disc_lift']})")
    print(f"  pairs where BOTH plain-Lap & RWSE tie  : {both_tie}")
    print(f"  EccLift separates where BOTH tie       : {ecclift_sep_where_both_tie}"
          f"   <== HEADLINE COUNT")
    print("=" * 72)

    verdict = (
        f"POSITIVE (tail-concentrated): EccLift separates {ecclift_sep_where_both_tie}"
        f" BREC pair(s) that BOTH plain-Lap AND RWSE tie -> strictly-new separation."
        if ecclift_sep_where_both_tie > 0 else
        "HONEST-NEGATIVE: EccLift separates NOTHING that plain-Lap/RWSE do not "
        "already separate on BREC."
    )
    print(verdict)
    print()
    print("Strictly-new pairs (EccLift separates; Lap & RWSE tie):")
    for r in ecclift_new_pairs:
        print(f"   category={r['category']:18s} pair_id={r['pair_id']:4d} "
              f"n={r['n0']}")

    out = {
        "generated": "full BREC separation audit for EccLift",
        "brec_pairs_total": n_pairs_total,
        "leak_guard_K": K,
        "totals": {
            "n_pairs": tot["n"],
            **{f"{k}_separates": tot[k] for k in encoders},
            "ecclift_leak_or_undefined": tot["ecclift_leak"],
            "both_Lap_and_RWSE_tie": both_tie,
            "ecclift_separates_where_both_tie": ecclift_sep_where_both_tie,
        },
        "per_category": per_cat,
        "strictly_new_pairs": ecclift_new_pairs,
        "verdict": verdict,
        "scope_note": ("SEPARATION / expressiveness only. No property-prediction "
                       "accuracy claim."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
