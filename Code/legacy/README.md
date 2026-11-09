# Legacy material

Scripts and figures from earlier versions of the paper, kept so that their
results remain reproducible. They are not used by the current (v3, AJP) paper.

| Path | Version | Purpose |
|---|---|---|
| `validation/run_validation.py` | v1.2 / v2.0 | Linear validation tables (series vs `expm`, modal, RK4, energy sign check) |
| `validation/generate_figures.py` | v1.2 | Linear validation figure (`figures/linear_validation.png`) |
| `validation/benchmark_structure.py` | v2.0 | Dense vs closed-form tridiagonal cost and accuracy; hanging-chain convergence |
| `validation/benchmark_symplectic.py` | v2.0 | O(n) Störmer–Verlet vs dense transition matrix |
| `validation/benchmark_nonlinear.py` | v2.0 | Exact O(n) nonlinear accelerations, RATTLE, small-amplitude limit |

Run them from the repository root, for example
`python Code/legacy/validation/benchmark_structure.py --quick`.

The corresponding manuscripts are archived as earlier versions of the Zenodo
record ([10.5281/zenodo.20102179](https://doi.org/10.5281/zenodo.20102179)) and
in the git tags `v1.2.0` and `v2.0.0`.
