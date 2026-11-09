# The Multiple Pendulum as a Beaded String

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20102179.svg)](https://doi.org/10.5281/zenodo.20102179)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Paper, figures and reference Python code for:

> **The multiple pendulum as a beaded string: hidden tridiagonal structure from normal modes to nonlinear dynamics**
> *Yonatan Ali Rodríguez Arias & Sandra Arellano González* — Instituto Politécnico Nacional (ESIME Zacatenco)

In the usual angle coordinates, the mass matrix of a planar pendulum with $n$ point masses is dense. The paper shows that this coupling is an artifact of the coordinates:

* **Factorization:** $\mathbf{M}=\mathbf{L}\mathbf{E}^{\mathsf T}\mathbf{D}\mathbf{E}\mathbf{L}$, with suffix masses $S_j=\sum_{i\ge j} m_i$.
* **Beaded string:** in horizontal displacements $\mathbf{x}=\mathbf{E}\mathbf{L}\boldsymbol\theta$, the small oscillations are those of a beaded string with tensions $gS_j$.
* **Tridiagonal formulas in angles:** closed forms for $\mathbf{M}^{-1}$ and the dynamical matrix $\mathbf{Q}=\mathbf{M}^{-1}\mathbf{K}$, computable in $O(n)$.
* **Normal modes:** a Jacobi matrix, so the frequencies are simple and mode $k$ has $k-1$ nodes. The spectrum converges to the hanging chain ($J_0$), and the discrete chain behaves like a continuous one of length $L+l/2$.
* **Exact nonlinear dynamics:** a tridiagonal rod-tension system whose zero-angle limit equals the inverse mass matrix, giving the accelerations in about ten lines of code and $O(n)$ operations.

The paper is written for *American Journal of Physics* readers and includes suggested problems.

---

## Repository structure

```text
Paper/
  Main.tex, refs.bib, figures/       manuscript source (one file, two builds)
  Main.pdf                           named version (preprint)
  submission_AJP/                    anonymous manuscript, cover letter, supplementary code
Code/
  src/structure.py    closed-form M^-1, Q, Jacobi matrix, normal modes (O(n))
  src/chain.py        tension system, exact nonlinear accelerations, RATTLE integrator
  src/symplectic.py   Störmer–Verlet for the linear problem, O(n) energy
  src/assembly.py     dense M, K and state-space matrix (reference)
  src/modal.py        generalized-eigenvalue modal solution (reference)
  src/nonlinear.py    dense nonlinear M(theta), c, g (reference)
  src/propagator.py   scaled-and-squared Taylor transition matrix (reference)
  tests/              40 automated tests of every formula
  validation/         figures_ajp.py (paper figures) and earlier benchmark scripts
```

## Quickstart

```python
import numpy as np
from src.structure import dynamic_matrix_tridiagonal, natural_frequencies
from src import chain

m = np.array([2.0, 1.5, 0.8]); l = np.array([1.2, 0.9, 0.6])

d, sub, sup = dynamic_matrix_tridiagonal(m, l)      # tridiagonal Q = M^-1 K, O(n)
omega = natural_frequencies(m, l)                   # simple frequencies, rad/s

theta = np.array([1.0, 0.5, -0.3]); thetadot = np.zeros(3)
acc = chain.angular_acceleration(theta, thetadot, m, l)   # exact nonlinear, O(n)
tau = chain.tensions(theta, thetadot, m, l)               # rod tensions
```

Run from the `Code/` directory, or install with `pip install -e ".[dev]"`.

## Tests and figures

```bash
pytest                                              # 40 tests
python Code/validation/figures_ajp.py               # Figs. 2 and 3 of the paper
```

The scripts `benchmark_structure.py`, `benchmark_symplectic.py` and `benchmark_nonlinear.py` reproduce the extended numerical studies of the earlier v2.0.0 preprint.

## Building the paper

```bash
cd Paper
pdflatex Main && bibtex Main && pdflatex Main && pdflatex Main            # named version
pdflatex "\def\anonymous{}\input{Main}"                                   # anonymous (AJP)
```

## Version history

* **v3.0.0:** rewritten as an AJP article around the beaded-string picture, with prior work credited (Gantmacher–Krein, Preston, Baraff, Bailey et al., Konishi & Yanagita).
* **v2.0.0:** closed-form tridiagonal structure and O(n) linear and nonlinear algorithms.
* **v1.x:** original operator framework.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff), or:

```bibtex
@misc{rodriguez_arellano_2026_pendulum,
  author = {Rodr{\'i}guez Arias, Yonatan Ali and Arellano Gonz{\'a}lez, Sandra},
  title  = {The multiple pendulum as a beaded string: hidden tridiagonal structure from normal modes to nonlinear dynamics},
  year   = {2026},
  doi    = {10.5281/zenodo.20102179},
  url    = {https://doi.org/10.5281/zenodo.20102179}
}
```
