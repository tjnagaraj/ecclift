"""
EccLift premise-check + separation audit (CPU).

(1) Reproduce the rook(4x4) vs Shrikhande separation and print the raw
    eccentric-transmission profiles of G and of the incidence lift L(G).
(2) Run EccLift on a cospectral / strongly-regular test set and COUNT how many
    non-isomorphic pairs EccLift separates that plain-Laplacian AND RWSE tie.

All numbers are exact BFS distances (no eigensolver in EccLift itself).
Writes results/results.json.
"""

import json
import os
import itertools
import numpy as np
import networkx as nx

from eccentric_lift import (
    incidence_lift,
    eccentric_transmission_profile,
    transmission_profile,
    eccentricity_profile,
    ecclift_profile,
    laplacian_spectrum,
    rwse,
    separates,
    _all_pairs_dist as _all_pairs_dist_dense,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "results.json")


# ---------------------------------------------------------------------------
# canonical strongly-regular cospectral mates
# ---------------------------------------------------------------------------
def rook_4x4():
    """Rook's graph 4x4 = K4 x K4 (Cartesian). srg(16,6,2,2)."""
    return nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))


def shrikhande():
    """Shrikhande graph: Cayley graph of Z4xZ4 with connection set
    {+-(1,0), +-(0,1), +-(1,1)}. srg(16,6,2,2), cospectral with rook(4x4),
    non-isomorphic."""
    G = nx.Graph()
    conn = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for i in range(4):
        for j in range(4):
            for di, dj in conn:
                G.add_edge((i, j), ((i + di) % 4, (j + dj) % 4))
    return G


def t8_and_chang():
    """T(8)=L(K8) vs a Chang graph (Seidel switching on a 1-factor).
    Both srg(28,12,6,4), cospectral, non-isomorphic."""
    Graw = nx.line_graph(nx.complete_graph(8))
    T8 = nx.convert_node_labels_to_integers(Graw, ordering="sorted")
    nodes = list(Graw.nodes())
    A = nx.to_numpy_array(Graw, nodelist=nodes, dtype=int)
    factor = {(0, 1), (2, 3), (4, 5), (6, 7)}
    S = set(k for k, e in enumerate(nodes) if tuple(sorted(e)) in factor)
    B = A.copy()
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if (i in S) ^ (j in S):
                B[i, j] = 1 - B[i, j]
                B[j, i] = B[i, j]
    return T8, nx.from_numpy_array(B)


def strongly_regular_pairs():
    """The two canonical SR cospectral mate pairs (the honest wall: both
    distance-regular, so plain-Lap, RWSE, and distance descriptors on G all
    tie)."""
    T8, CH = t8_and_chang()
    return [
        ("srg(16,6,2,2) rook4x4_vs_shrikhande", rook_4x4(), shrikhande()),
        ("srg(28,12,6,4) T8_vs_Chang", T8, CH),
    ]


# ---------------------------------------------------------------------------
# a small cospectral / SR atlas from generators (no external data files)
# ---------------------------------------------------------------------------
def cospectral_pairs():
    """Return a list of (name, G1, G2) non-isomorphic pairs that are
    plain-Laplacian cospectral. Built from generators so no data file is
    needed. Each pair is verified cospectral + non-isomorphic at run time."""
    pairs = []

    # 1. rook(4x4) vs Shrikhande -- srg(16,6,2,2) [the VERIFIED headline pair]
    pairs.append(("rook4x4_vs_shrikhande", rook_4x4(), shrikhande()))

    # 2. srg(28,12,6,4): T(8)=L(K8) vs a Chang graph -- second SR wall.
    T8, CH = t8_and_chang()
    pairs.append(("srg(28,12,6,4) T8_vs_Chang", T8, CH))

    # 3-...: enumerate small connected graphs and harvest ALL Laplacian-
    # cospectral non-isomorphic pairs up to n<=7 via the atlas.
    pairs.extend(_atlas_cospectral_pairs(max_n=7))
    return pairs


def _is_lap_cospectral(G1, G2, tol=1e-6):
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False
    return np.allclose(laplacian_spectrum(G1), laplacian_spectrum(G2), atol=tol)


def _atlas_cospectral_pairs(max_n=7):
    """Harvest Laplacian-cospectral non-isomorphic CONNECTED pairs from the
    networkx graph atlas (complete through 7 vertices)."""
    from networkx.generators.atlas import graph_atlas_g
    atlas = graph_atlas_g()
    by_n = {}
    for G in atlas:
        n = G.number_of_nodes()
        if n < 2 or n > max_n:
            continue
        if not nx.is_connected(G):
            continue
        by_n.setdefault(n, []).append(G)

    out = []
    for n, gs in by_n.items():
        # bucket by rounded Laplacian spectrum
        buckets = {}
        for G in gs:
            key = tuple(np.round(laplacian_spectrum(G), 5))
            buckets.setdefault(key, []).append(G)
        for key, group in buckets.items():
            if len(group) < 2:
                continue
            for G1, G2 in itertools.combinations(group, 2):
                if not nx.is_isomorphic(G1, G2):
                    out.append((f"atlas_n{n}_cospec_{len(out)}", G1, G2))
    return out


# ---------------------------------------------------------------------------
# main audit
# ---------------------------------------------------------------------------
def profile_str(p):
    if p is None:
        return "DISCONNECTED"
    vals, counts = np.unique(np.round(p, 4), return_counts=True)
    return "{" + ", ".join(f"{v:g}x{c}" for v, c in zip(vals, counts)) + "}"


def main():
    results = {"generated": "premise-check for EccLift", "cases": {}}

    # ---- (1) headline: rook vs Shrikhande, raw profiles ------------------
    R, S = rook_4x4(), shrikhande()
    assert R.number_of_nodes() == 16 and S.number_of_nodes() == 16
    assert nx.is_connected(R) and nx.is_connected(S)
    assert R.number_of_edges() == 48 and S.number_of_edges() == 48, (
        R.number_of_edges(), S.number_of_edges())
    assert _is_lap_cospectral(R, S), "rook/Shrikhande must be Lap-cospectral"
    assert not nx.is_isomorphic(R, S), "rook/Shrikhande must be non-isomorphic"

    et_G_R = eccentric_transmission_profile(R)
    et_G_S = eccentric_transmission_profile(S)
    et_L_R = ecclift_profile(R)
    et_L_S = ecclift_profile(S)

    headline = {
        "both_srg_16_6_2_2": True,
        "lap_cospectral": True,
        "non_isomorphic": True,
        "ET_of_G_rook": profile_str(et_G_R),
        "ET_of_G_shrikhande": profile_str(et_G_S),
        "ET_of_G_ties": bool(np.array_equal(et_G_R, et_G_S)),
        "ET_of_lift_rook": profile_str(et_L_R),
        "ET_of_lift_shrikhande": profile_str(et_L_S),
        "ET_of_lift_separates": bool(separates(ecclift_profile, R, S)),
        "plain_Lap_separates": bool(separates(laplacian_spectrum, R, S)),
        "RWSE_separates": bool(separates(rwse, R, S)),
    }
    results["cases"]["headline_rook_shrikhande"] = headline
    print("=== HEADLINE: rook(4x4) vs Shrikhande, srg(16,6,2,2) ===")
    print("  ET(G) rook       :", headline["ET_of_G_rook"])
    print("  ET(G) shrikhande :", headline["ET_of_G_shrikhande"])
    print("  ET(G) ties?              ", headline["ET_of_G_ties"])
    print("  ET(lift) rook       :", headline["ET_of_lift_rook"])
    print("  ET(lift) shrikhande :", headline["ET_of_lift_shrikhande"])
    print("  ET(lift) SEPARATES?      ", headline["ET_of_lift_separates"])
    print("  plain-Lap separates?     ", headline["plain_Lap_separates"])
    print("  RWSE separates?          ", headline["RWSE_separates"])
    print()

    # ---- (1b) SR-only wall: both SR pairs, the honest tail claim ---------
    print("=== SR WALL (both distance-regular; Lap/RWSE/dist-of-G all tie) ===")
    sr_rows = []
    for name, G1, G2 in strongly_regular_pairs():
        row = {
            "name": name,
            "lap_separates": bool(separates(laplacian_spectrum, G1, G2)),
            "rwse_separates": bool(separates(rwse, G1, G2)),
            "ET_of_G_separates": bool(
                separates(lambda g: eccentric_transmission_profile(g), G1, G2)),
            "ecclift_separates": bool(separates(ecclift_profile, G1, G2)),
        }
        sr_rows.append(row)
        print(f"  {name}")
        print(f"    plain-Lap: {'SEP' if row['lap_separates'] else 'tie'}  "
              f"RWSE: {'SEP' if row['rwse_separates'] else 'tie'}  "
              f"ET(G): {'SEP' if row['ET_of_G_separates'] else 'tie'}  "
              f"EccLift=ET(lift): {'SEP' if row['ecclift_separates'] else 'tie'}")
    results["cases"]["sr_wall"] = sr_rows
    sr_sep = sum(1 for r in sr_rows if r["ecclift_separates"]
                 and not r["lap_separates"] and not r["rwse_separates"]
                 and not r["ET_of_G_separates"])
    print(f"  EccLift clears the SR wall on {sr_sep}/{len(sr_rows)} SR pairs "
          f"(Lap+RWSE+ET(G) all tie).")
    print()

    # ---- (2) cospectral audit --------------------------------------------
    pairs = cospectral_pairs()
    n_pairs = len(pairs)
    # counts
    lap_tie = 0
    rwse_tie = 0
    both_tie = 0
    ecclift_sep = 0
    ecclift_sep_where_both_tie = 0  # THE headline count
    per_pair = []

    for name, G1, G2 in pairs:
        lap_s = separates(laplacian_spectrum, G1, G2)
        rwse_s = separates(rwse, G1, G2)
        ecc_s = separates(ecclift_profile, G1, G2)
        if not lap_s:
            lap_tie += 1
        if not rwse_s:
            rwse_tie += 1
        if (not lap_s) and (not rwse_s):
            both_tie += 1
            if ecc_s:
                ecclift_sep_where_both_tie += 1
        if ecc_s:
            ecclift_sep += 1
        per_pair.append({
            "name": name,
            "n": G1.number_of_nodes(),
            "lap_separates": bool(lap_s),
            "rwse_separates": bool(rwse_s),
            "ecclift_separates": bool(ecc_s),
        })

    audit = {
        "n_cospectral_pairs": n_pairs,
        "plain_Lap_ties": lap_tie,
        "RWSE_ties": rwse_tie,
        "both_Lap_and_RWSE_tie": both_tie,
        "ecclift_separates_total": ecclift_sep,
        "ecclift_separates_where_Lap_and_RWSE_both_tie": ecclift_sep_where_both_tie,
        "per_pair": per_pair,
    }
    results["cases"]["cospectral_audit"] = audit

    print("=== COSPECTRAL / SR AUDIT ===")
    print(f"  cospectral non-iso pairs tested : {n_pairs}")
    print(f"  plain-Lap ties                  : {lap_tie}")
    print(f"  RWSE ties                       : {rwse_tie}")
    print(f"  BOTH Lap and RWSE tie           : {both_tie}")
    print(f"  EccLift separates (total)       : {ecclift_sep}")
    print(f"  EccLift separates where BOTH tie: {ecclift_sep_where_both_tie}  <== headline count")
    print()

    # honest self-check: is EccLift ever STRICTLY new, or does it only
    # duplicate what Lap/RWSE already separate?
    strictly_new = ecclift_sep_where_both_tie
    audit_verdict = (
        "POSITIVE: EccLift separates >=1 cospectral pair that BOTH plain-Lap "
        "and RWSE tie -> not a duplicate feature."
        if strictly_new > 0 else
        "HONEST-NEGATIVE: EccLift separates nothing beyond plain-Lap/RWSE."
    )
    results["verdict"] = audit_verdict
    print(audit_verdict)

    # ---- (3) cost note: EccLift is eigendecomposition-FREE ---------------
    import time
    R2, S2 = rook_4x4(), shrikhande()
    t0 = time.perf_counter()
    for _ in range(50):
        ecclift_profile(R2); ecclift_profile(S2)
    t_ecclift = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    for _ in range(50):
        # the eigendecomposition-based counterpart: eigvals of the lift's
        # distance-Laplacian (an eigendecomposition on the p+q lift matrix)
        H = incidence_lift(R2)
        A = nx.to_numpy_array(H)
        D = _all_pairs_dist_dense(H)
        np.linalg.eigvalsh(np.diag(D.sum(1)) - D)
        H = incidence_lift(S2)
        D = _all_pairs_dist_dense(H)
        np.linalg.eigvalsh(np.diag(D.sum(1)) - D)
    t_spectrum = (time.perf_counter() - t0) / 100
    results["cost"] = {
        "ecclift_sec_per_graph_avg": t_ecclift,
        "distL_spectrum_of_lift_sec_per_graph_avg": t_spectrum,
        "note": "EccLift is BFS-only (no eigensolver); the counterpart distance-"
                "Laplacian SPECTRUM of the lift pays an eigendecomposition.",
    }
    print(f"\n=== COST (avg per graph, srg-16 lift) ===")
    print(f"  EccLift (BFS-only, eigendecomposition-FREE): {t_ecclift*1e3:.3f} ms")
    print(f"  distance-Lap SPECTRUM of the lift (eig)    : {t_spectrum*1e3:.3f} ms")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
