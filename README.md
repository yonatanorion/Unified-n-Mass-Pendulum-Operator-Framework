# A Unified Structural and Operator Framework for the Linear n-Mass Pendulum

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20102179.svg)](https://doi.org/10.5281/zenodo.20102179)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

This repository contains the scientific paper, figures, and reference Python implementation for:

> **A Unified Structural and Operator Framework for the Linear $n$-Mass Pendulum**  
> *Yonatan Ali Rodríguez Arias & Sandra Arellano González*

The project presents:

* **Direct Structural Matrix Assembly:** $\mathcal{O}(n^2)$ formulation via the **cumulative mass-sum rule** ($S_j = \sum_{i=j}^n m_i$).
* **Linear State-Space Generation:** Compact first-order formulation $\dot{\mathbf{R}} = \mathcal{A}\mathbf{R}$ with diagonal gravitational stiffness.
* **Exact Modal Decoupling:** Generalized eigenvalue solution ($\mathbf{K}\mathbf{U} = \mathbf{M}\mathbf{U}\mathbf{\Omega}^2$) providing a reference baseline.
* **Operator-Based Linear Propagation:** Scaled-and-squared **Peano-Baker / Taylor series** approximation for the discrete transition matrix $\Phi(\Delta t) = e^{\mathcal{A}\Delta t}$.
* **Rigorous Numerical Validation:** Concordance to double-precision machine epsilon ($\sim 10^{-15}$ rad) against `scipy.linalg.expm` and modal solutions.
* **Nonlinear Double-Pendulum Energy Check:** Numerical validation confirming the corrected coupling sign $\sin(\theta_d - \theta_k)$ and energy conservation.

---

# Quickstart

```python
import numpy as np
from src.assembly import assemble_system
from src.propagator import peano_baker_step, propagate_linear

# 1. Define n-mass pendulum parameters (e.g., n = 3)
masses = [2.0, 1.5, 0.8]   # kg
lengths = [1.2, 0.9, 0.6]  # m

# 2. Assemble system matrices in O(n^2)
A, M, K = assemble_system(masses, lengths)

# 3. Precompute discrete-time transition operator
dt = 0.01  # s
Phi = peano_baker_step(A, dt, N=20)

# 4. Propagate linear dynamics from initial displacement
t = np.arange(0, 10.0 + dt, dt)
R0 = np.array([0.1, 0.05, 0.02, 0.0, 0.0, 0.0])  # [theta0, omega0]
trajectory = propagate_linear(Phi, R0, t)

print(f"Simulated {len(t)} time steps with shape: {trajectory.shape}")
```

---

# Repository Structure

```text
.
├── Code/
│   ├── figures/
│   │   └── linear_validation.png
│   ├── src/
│   │   ├── __init__.py
│   │   ├── assembly.py
│   │   ├── modal.py
│   │   ├── nonlinear.py
│   │   └── propagator.py
│   ├── SymbolicGenTester&Examples/
│   │   ├── N-MassEqSymboGen.py
│   │   ├── Output_Equations_pendulum_5_masses.md
│   │   ├── Output_Equations_pendulum_5_masses.pdf
│   │   └── README.md
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_assembly.py
│   │   ├── test_modal.py
│   │   ├── test_nonlinear.py
│   │   └── test_propagator.py
│   ├── validation/
│   │   ├── generate_figures.py
│   │   └── run_validation.py
│   └── setup.py
├── Paper/
│   ├── figures/
│   │   ├── 4mass_pendulum.png
│   │   ├── linear_validation.png
│   │   └── n_mass_pendulum.png
│   ├── Main.tex
│   └── Main.pdf
├── output/
│   └── pdf/
│       └── Unified_n_Mass_Pendulum_revised.pdf
├── pyproject.toml
├── LICENSE
└── README.md
```

---

# Modules

## `src/assembly.py`

Implements the linear matrix assembly using the cumulative mass-sum rule:
* Mass matrix ($\mathbf{M}$)
* Diagonal gravitational stiffness matrix ($\mathbf{K}$)
* State-space operator ($\mathcal{A}$)

Core formulation:

$$
M_{jk} = \left(\sum_{i=\max(j,k)}^n m_i\right) l_j l_k, \qquad K_{jj} = \left(\sum_{i=j}^n m_i\right) g l_j
$$

---

## `src/propagator.py`

Implements operator-based linear propagation using a scaled-and-squared Peano-Baker/Taylor expansion:

$$
\mathbf{B} = \frac{\mathcal{A}\Delta t}{2^s}, \qquad \Phi(\Delta t) \approx \left(\sum_{k=0}^{N}\frac{\mathbf{B}^k}{k!}\right)^{2^s}
$$

Scaling controls truncation error when $\|\mathcal{A}\Delta t\|$ is large. The transition matrix $\Phi(\Delta t)$ is computed once and reused across simulations.

---

## `src/modal.py`

Implements the standard generalized-eigenvalue solution of $\mathbf{M}\ddot{\boldsymbol{\theta}}+\mathbf{K}\boldsymbol{\theta}=\mathbf{0}$ via `scipy.linalg.eigh`, providing an independent reference solution.

---

## `src/nonlinear.py`

Implements the exact nonlinear mass matrix $\mathbf{M}(\boldsymbol{\theta})$, velocity-squared coupling vector $\mathbf{c}(\boldsymbol{\theta}, \dot{\boldsymbol{\theta}})$, gravity vector $\mathbf{g}(\boldsymbol{\theta})$, first-order state derivative, and total physical energy $E = T + U$.

---

## `SymbolicGenTester&Examples/N-MassEqSymboGen.py`

CLI tool and script to generate full symbolic equations of motion for an arbitrary $n$-mass pendulum formatted in Markdown and LaTeX.

```bash
# Direct CLI execution for 5 masses:
python "Code/SymbolicGenTester&Examples/N-MassEqSymboGen.py" -n 5 -o output_5_mass.md
```

---

# Installation & Testing

Install dependencies in development mode:

```bash
pip install -e ".[dev]"
```

Run unit tests locally:

```bash
pytest
```

or with `uv`:

```bash
uv run pytest
```

---

# Running Validations & Figures

## Run numerical validation tables

```bash
python Code/validation/run_validation.py
```

## Generate paper figures

```bash
python Code/validation/generate_figures.py
```

---

# License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

# Citation

If you use this repository or framework in your research, please cite:

```bibtex
@article{rodriguez_arellano_2026_pendulum,
  author    = {Rodr{\'i}guez Arias, Yonatan Ali and Arellano Gonz{\'a}lez, Sandra},
  title     = {A Unified Structural and Operator Framework for the Linear $n$-Mass Pendulum},
  year      = {2026},
  doi       = {10.5281/zenodo.20102179},
  url       = {https://doi.org/10.5281/zenodo.20102179}
}
```

