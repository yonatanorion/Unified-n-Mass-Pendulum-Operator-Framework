# A Unified Structural and Operator Framework for the Linear n-Mass Pendulum

This repository contains the reference Python implementation accompanying the manuscript:

> **A Unified Structural and Operator Framework for the Linear n-Mass Pendulum**

The project presents:

* Structural matrix assembly via the **cumulative mass-sum rule**
* Linear state-space generation for arbitrary (n)-mass pendulums
* Standard modal decoupling through a generalized eigenvalue problem
* Operator-based propagation using a scaled-and-squared **Peano-Baker/Taylor series**
* Numerical validation against the exact matrix exponential
* A nonlinear double-pendulum energy check for the corrected coupling sign

---

# Repository Structure

```text
REPOSITORY PAPER/
│
├── Code/
│   │
│   ├── figures/
│   │   └── linear_validation.png
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── assembly.py
│   │   ├── modal.py
│   │   ├── nonlinear.py
│   │   └── propagator.py
│   │
│   ├── SymbolicGenTester&Examples/
│   │   ├── N-MassEqSymboGen.py
│   │   ├── Output_Equations_pendulum_5_masses.md
│   │   ├── Output_Equations_pendulum_5_masses.pdf
│   │   └── README.md
│   │
│   ├── validation/
│   │   ├── generate_figures.py
│   │   └── run_validation.py
│   │
│   └── setup.py
│
├── Paper/
│   │
│   ├── figures/
│   │
│   ├── Main.tex
│   │
│   └── RodriguezArias_ArellanoGonzalez_UnifiedFramework_NMassPendulum_JTCAM.pdf.pdf
│
├── .gitignore
│
└── README.md
```

---

# Modules

## `src/assembly.py`

Implements the linear matrix assembly using the cumulative mass-sum rule:

* Mass matrix (M)
* Gravitational stiffness matrix (K)
* State-space operator (A)

Core formulation:

$$
M_{jk} =
\left(
\sum_{i=\max(j,k)}^n m_i
\right)
l_j l_k
$$

---

## `src/propagator.py`

Implements operator-based linear propagation using a scaled-and-squared Peano-Baker/Taylor expansion:

$$
B=\frac{A\Delta t}{2^s}, \qquad
\Phi(\Delta t)\approx
\left(\sum_{k=0}^{N}\frac{B^k}{k!}\right)^{2^s}
$$

Scaling controls the truncation error when $\lVert A\Delta t\rVert$ is not small. The transition matrix is precomputed once and reused efficiently across simulations.

---

## `src/modal.py`

Implements the standard generalized-eigenvalue solution of
$M\ddot{\theta}+K\theta=0$ and provides an independent reference for the
state-space propagator.

## `src/nonlinear.py`

Implements the exact nonlinear mass matrix, velocity-squared vector, gravity
vector, state derivative, and total energy. The velocity vector uses
$\sin(\theta_d-\theta_k)$ without an additional index-dependent sign.

---

## `validation/run_validation.py`

Reproduces the numerical tables from the manuscript:

* Structural scaling benchmarks
* Linear propagation validation against `scipy.linalg.expm` and modal decoupling
* RK4 consistency comparison
* Nonlinear energy conservation and sign-regression check

---

## `validation/generate_figures.py`

Generates manuscript figures, including:

* Time-domain trajectory comparisons
* Absolute propagation error plots

Output:

```text
figures/linear_validation.png
```

---

# Installation

Recommended environment:

```bash
pip install numpy scipy matplotlib
```

---

# Usage

## Run numerical validations

```bash
python Code/validation/run_validation.py
```

## Generate figures

```bash
python Code/validation/generate_figures.py
```

---

# Reproducibility

The repository is designed to reproduce the numerical experiments reported in the manuscript.

Reference solutions use:

```python
scipy.linalg.expm
```

for exact matrix exponential propagation.

---

# License

This repository is released under the MIT License.

---

# Citation

If you use this repository, please cite:

Rodríguez Arias, Y. A., & Arellano González, S.
"A Unified Structural and Operator Framework for the Linear n-Mass Pendulum"

Concept DOI: https://doi.org/10.5281/zenodo.20102179

Version DOI used by the manuscript: https://doi.org/10.5281/zenodo.20102180

