"""
EccLift: the eccentric-transmission profile of the incidence lift as a cheap,
eigendecomposition-FREE, two-level structural positional encoding.

Self-contained definitions (expressiveness / PE terminology; see the paper for
the graph-theory references).

-------------------------------------------------------------------------------
Two parameter-free per-node scalars on a connected graph H = (V, E):

  transmission          t_H(v)   = sum_{u in V} d_H(v, u)
  eccentricity          ecc_H(v) = max_{u in V} d_H(v, u)
  eccentric-transmission et_H(v) = ecc_H(v) * t_H(v)

where d_H is the shortest-path (hop) distance. The ECCENTRIC-TRANSMISSION
PROFILE of H is the sorted multiset  ET(H) = { et_H(v) : v in V }.  It is a
permutation-invariant, training-free, eigendecomposition-FREE graph encoding
(pure BFS: O(|V|*|E|), no eigensolver).

-------------------------------------------------------------------------------
The INCIDENCE LIFT of G is the bipartite two-level graph L(G):
  * one "vertex-node" per vertex of G     (the 0-cells, size p = |V(G)|)
  * one "edge-node"   per edge   of G     (the 1-cells, size q = |E(G)|)
  * a vertex-node u is joined to an edge-node e iff u is an endpoint of e in G.
This is the subdivision / node-hyperedge incidence structure S(G), i.e. the
classical subdivision graph (Harary, Graph Theory, 1969).

EccLift(G) := ET( L(G) ) -- the eccentric-transmission profile computed ON the
incidence lift. This composition is the whole idea: eccentric-transmission of G
COLLAPSES on regular graphs (all vertices share ecc and transmission), so ET(G)
is degenerate for regular G; but L(G) is NOT regular even when G is (vertex-
nodes have degree deg_G(u), edge-nodes have degree 2), so ET(L(G)) breaks the
degeneracy AT NO EIGENDECOMPOSITION COST.
"""

import itertools
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# the incidence lift  L(G)  (subdivision / node-hyperedge incidence structure)
# ---------------------------------------------------------------------------
def incidence_lift(G):
    """Return the incidence lift L(G) as a networkx.Graph.
    Vertex-nodes: ('v', u). Edge-nodes: ('e', frozenset({a,b})).
    Vertex-node ('v',u) ~ edge-node ('e',{a,b}) iff u in {a,b}."""
    H = nx.Graph()
    for u in G.nodes():
        H.add_node(("v", u))
    for a, b in G.edges():
        e = ("e", frozenset((a, b)))
        H.add_node(e)
        H.add_edge(("v", a), e)
        H.add_edge(("v", b), e)
    return H


# ---------------------------------------------------------------------------
# parameter-free per-node scalars + the eccentric-transmission profile
# ---------------------------------------------------------------------------
def _all_pairs_dist(H):
    n = H.number_of_nodes()
    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    D = np.full((n, n), np.inf)
    for src, lengths in nx.all_pairs_shortest_path_length(H):
        i = idx[src]
        for dst, d in lengths.items():
            D[i, idx[dst]] = d
    return D


def eccentric_transmission_profile(H):
    """ET(H): sorted multiset of ecc(v)*transmission(v). Returns None if H is
    disconnected (distances undefined)."""
    if H.number_of_nodes() == 0 or not nx.is_connected(H):
        return None
    D = _all_pairs_dist(H)
    transmission = D.sum(axis=1)
    eccentricity = D.max(axis=1)
    et = eccentricity * transmission
    return np.sort(et)


def transmission_profile(H):
    if not nx.is_connected(H):
        return None
    D = _all_pairs_dist(H)
    return np.sort(D.sum(axis=1))


def eccentricity_profile(H):
    if not nx.is_connected(H):
        return None
    D = _all_pairs_dist(H)
    return np.sort(D.max(axis=1))


def ecclift_profile(G):
    """EccLift(G) = ET(L(G)): eccentric-transmission profile of the incidence
    lift. The full two-level PE. Eigendecomposition-FREE."""
    return eccentric_transmission_profile(incidence_lift(G))


# ---------------------------------------------------------------------------
# baseline (eigendecomposition-BASED) encodings we separate against
# ---------------------------------------------------------------------------
def laplacian_spectrum(G):
    A = nx.to_numpy_array(G)
    L = np.diag(A.sum(1)) - A
    return np.sort(np.linalg.eigvalsh(L))


def rwse(G, k=8):
    """Random-walk structural encoding: sorted diagonal return probabilities of
    P, P^2, ..., P^k (a standard structural PE)."""
    A = nx.to_numpy_array(G)
    d = A.sum(1)
    d[d == 0] = 1.0
    P = A / d[:, None]
    diags = []
    M = np.eye(P.shape[0])
    for _ in range(k):
        M = M @ P
        diags.append(np.sort(np.diag(M).copy()))
    return np.concatenate(diags)


def _profiles_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=tol)


def separates(encoder, G1, G2, tol=1e-6):
    """True iff `encoder` gives DIFFERENT profiles on G1, G2 (i.e. tells them
    apart)."""
    return not _profiles_equal(encoder(G1), encoder(G2), tol=tol)
