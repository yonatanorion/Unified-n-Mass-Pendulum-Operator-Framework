# A Unified Structural and Operator Framework for the Linear n-Mass Pendulum

This repository contains the reference Python implementation accompanying the manuscript:

> **A Unified Structural and Operator Framework for the Linear n-Mass Pendulum**

The project presents:

* Structural matrix assembly via the **cumulative mass-sum rule**
* Linear state-space generation for arbitrary (n)-mass pendulums
* Operator-based propagation using a truncated **Peano–Baker series**
* Numerical validation against the exact matrix exponential

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

$[
M_{jk} =
\left(
\sum_{i=\max(j,k)}^n m_i
\right)
l_j l_k
]$

---

## `src/propagator.py`

Implements operator-based linear propagation using the truncated Peano–Baker expansion:

$[
\Phi(\Delta t)=
\sum_{k=0}^{N}
\frac{(A\Delta t)^k}{k!}
]$

The transition matrix is precomputed once and reused efficiently across simulations.

---

## `validation/run_validation.py`

Reproduces the numerical tables from the manuscript:

* Structural scaling benchmarks
* Linear propagation validation
* RK4 comparison

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
python validation/run_validation.py
```

## Generate figures

```bash
python validation/generate_figures.py
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

DOI:
https://zenodo.org/records/20102180
The reference implementation and validation scripts are publicly archived at Zenodo:





