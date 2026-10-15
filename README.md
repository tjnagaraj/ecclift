# EccLift: Separating Strongly Regular Cospectral Mates without an Eigensolver

Code for the paper published in the Proceedings of the Fifth Learning on Graphs Conference (LoG 2026).

**Authors:** Nagaraj Thenkarai Janakiraman, Vishwashankar Thenkarai Janakiraman, Janakiraman Thenkarai Nagarajan, Mahemaa Rajasekaran, Sruthikeerthi Nandita Rajagopalakrishnan

## Overview

EccLift is a parameter-free, permutation-invariant, eigendecomposition-free graph-level
separation invariant: the sorted multiset of eccentricity times transmission,
`ecc(w) * t(w)`, over all nodes `w` of the incidence lift (subdivision graph) of `G`,
computed by breadth-first search alone. It separates the strongly-regular cospectral
mates rook(4x4)/Shrikhande and T(8)/Chang, on which the Laplacian spectrum, RWSE and the
eccentric transmission of `G` itself all tie, and across BREC it separates 56 pairs that
both the Laplacian spectrum and RWSE miss. The paper also reduces EccLift exactly to two
distance profiles of `G`, which fixes its failure boundary (it provably ties on all 20
BREC Distance-Regular pairs) while it still separates the same-array pair Doob D(1,1) vs
Hamming H(3,4). EccLift is designed for separation; the paper makes no downstream-accuracy
claims.

## Setup

- Python >= 3.9 (tested with Python 3.13, NumPy 2.5, NetworkX 3.7).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- **BREC data (needed only for `brec_audit.py`, Part C of `characterization.py`, and the
  BREC scan in `converse_explore.py`).** The BREC dataset is not bundled and the scripts do
  not download it. Get `BREC_data_all.zip` from the BREC repository
  (<https://github.com/GraphPKU/BREC>, see its "Data Preparation" section), unzip it, and
  place `brec_v3.npy` at

  ```
  data/BREC/brec_v3.npy
  ```

  relative to the repository root, or point the `BREC_PATH` environment variable at the
  file. Without the file, `characterization.py` and `converse_explore.py` run and skip
  their BREC parts; `brec_audit.py` exits with a message.

## Layout

```
code/eccentric_lift.py      the encoding: incidence lift, ET profile, EccLift, baselines
code/audit.py               witness pairs + small cospectral atlas
code/characterization.py    distance-geometry reduction, DR boundary, Doob/Hamming, prior-art check
code/converse_explore.py    counterexample search for the conjecture
code/small_order_completeness.py  EccLift completeness on all connected graphs of order 2..7
code/brec_audit.py          leak-guarded BREC separation audit
code/runtime_scaling.py     runtime/memory scaling
results/                    committed outputs backing the numbers in the paper
```

All scripts write into `results/` and can be run from the repository root.

## Scripts and what they back in the paper

| Script | What it computes | Paper location | Output (`results/`) |
|---|---|---|---|
| `code/eccentric_lift.py` | Library: incidence lift `L(G)`, eccentric-transmission profile `ET`, `EccLift(G) = ET(L(G))`, and the Laplacian-spectrum / RWSE (k=8) baselines. Imported by all other scripts. | Definitions (Sec. 2); Algorithm 1 (Sec. 4); worked example `EccLift(P_3) = {12,21,21,40,40}` (Fig. 1, Sec. 4) | (none; library) |
| `code/audit.py` | Rook(4x4)/Shrikhande raw `ET(G)` and `ET(L(G))` profiles; T(8)/Chang; Laplacian / RWSE / `ET(G)` ties on both strongly-regular pairs; all Laplacian-cospectral connected atlas pairs up to 7 vertices; per-graph timing of EccLift vs. the lift's distance-Laplacian spectrum. | Prop. "Strongly-regular wall, via the edge profile" (`{1080^16, 1452^48}` vs `{1080^16, 1440^48}`); Table 1 (`tab:witness`), rows srg(16,6,2,2) and srg(28,12,6,4); Sec. 4 "Complexity" (2.5 vs 2.9 ms/graph); Sec. 5 "Negative results" (2) (RWSE already separates the small atlas pairs) | `results.json` |
| `code/characterization.py` | Part A: exact lift-distance identities; Part B: vertex distance-distribution / edge-min-distance profile (emdp) on rook/Shrikhande (28 vs 29 edges at edge-distance 1) and the profile-vs-EccLift check over 6,444 pairs; Part C: all 20 BREC Distance-Regular pairs (same intersection array, equal emdp, EccLift ties; needs BREC); Part D: distance and Aouchiche-Hansen distance-Laplacian spectra on the strongly-regular pairs; Part E: Doob D(1,1) vs Hamming H(3,4). | Sec. 3: Lemma "Lift distances", Prop. "Strongly-regular wall", Prop. "Failure boundary on BREC Distance-Regular" (20/20), Prop. "A same-array distance-regular pair EccLift separates" (`{12096^64, 15024^192, 15088^96}` vs `{12096^64, 15088^288}`), Remark "Prior art: not a distance spectrum"; Table 1 (`tab:witness`), rows Doob/Hamming and BREC Dist.-Reg. | `results_characterization.json` |
| `code/converse_explore.py` | Exact-integer counterexample search for the conjecture: all 853 connected graphs of order 7 (no two share an EccLift value); every BREC pair on which EccLift ties (143: 90 CFI, 20 Distance-Regular, 20 4-Vertex, 11 Extension, 2 Regular; needs BREC); a 12,704-graph regular/circulant/Cartesian/tensor/random pool with 29 non-isomorphic EccLift-tie pairs. Zero counterexamples. | Conjecture "The scalar-product collapse loses no separating power" and the paragraph after it; Sec. 1 Contributions; Sec. 5 "Negative results" (1) (the missed Regular pairs have coinciding profiles); Conclusion | `converse_explore_out.json` |
| `code/small_order_completeness.py` | Enumerates all connected graphs of order 2..7 from the NetworkX graph atlas (`graph_atlas_g`, one graph per isomorphism class), computes EccLift with `eccentric_lift.py`, and counts collisions (pairs of non-isomorphic graphs with equal EccLift), per order and pooled across orders: 995 graphs (1/2/6/21/112/853), 995 distinct values, 0 collisions. | Sec. 3, paragraph after Conjecture "The scalar-product collapse loses no separating power" ("EccLift is a *complete* invariant on all 995 connected graphs of order <= 7 (no two non-isomorphic such graphs share an EccLift value)"); the same claim in the Abstract, Sec. 1 Contributions (second item), the paragraph "Why EccLift rather than RowLift", Sec. 6 "When to use it" (ii), and the Conclusion | `results_small_order_completeness.json`, `results_small_order_completeness.txt` |
| `code/brec_audit.py` | Leak-guarded per-category BREC audit of plain Laplacian spectrum, RWSE, `ET(G)` and EccLift over the 400 pairs; both-tie counts; strictly-new separations (needs BREC). | Table 2 (`tab:brec`, "BREC separation audit"): 247 EccLift separations, 187 both-tie, 56 strictly new (49 Regular, 7 CFI), 10 excluded pairs; Table 1 (`tab:witness`), rows BREC Regular / BREC CFI; Sec. 1 and Abstract | `results_brec.json` |
| `code/runtime_scaling.py` | Median wall time (3 runs) and peak Python-heap memory of `ecclift_profile` on random r-regular graphs, r in {3, 8}, n = 100 ... 3200, seed 7, plus the empirical exponent in `|E|` between successive sizes. | Appendix A, Table 3 (`tab:runtime`, "EccLift runtime/memory scaling"); Sec. 4 "Complexity" (exponent 2.07-2.27) | `results_runtime_scaling.json`, `results_runtime_scaling.txt` |

Notes:
- All EccLift counts are exact-integer BFS computations. Wall-clock times (in
  `results.json` "cost" and in the runtime-scaling files) are machine-dependent and will
  differ from the committed values.
- The RWSE column of Table 2 uses floating-point return probabilities rounded to 1e-5;
  as noted in the table caption, one borderline pair can move between the Basic and
  Extension cells across BLAS builds (57/91 vs. 58/90) without affecting any EccLift or
  both-tie count.

## How to reproduce

From the repository root:

```bash
# no external data needed
python code/audit.py               # ~1 s     -> results/results.json
python code/characterization.py    # ~10 s without BREC (Part C needs BREC) -> results/results_characterization.json
python code/small_order_completeness.py  # ~1 s -> results/results_small_order_completeness.{json,txt}
python code/converse_explore.py    # ~35 s without BREC (order-7 search + 12,704-graph pool); BREC scan adds time
python code/runtime_scaling.py     # ~10+ min, peak memory ~2 GB at n=3200, r=8

# need data/BREC/brec_v3.npy (see Setup)
python code/brec_audit.py          # -> results/results_brec.json
```

Runtimes are approximate, single CPU thread. Running a script overwrites the
corresponding committed file in `results/`; use `git diff results/` to compare.

## Citation

```bibtex
@InProceedings{janakiraman2026ecclift,
  title     = {{EccLift}: Separating Strongly Regular Cospectral Mates without an Eigensolver},
  author    = {Janakiraman, Nagaraj Thenkarai and Janakiraman, Vishwashankar Thenkarai and Nagarajan, Janakiraman Thenkarai and Rajasekaran, Mahemaa and Rajagopalakrishnan, Sruthikeerthi Nandita},
  booktitle = {Proceedings of the Fifth Learning on Graphs Conference},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```

The PMLR volume and page numbers will be added when the proceedings are published.

## License

MIT (see [LICENSE](LICENSE)).

## Disclaimer

**Disclaimer:** This is a personal project. The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.
