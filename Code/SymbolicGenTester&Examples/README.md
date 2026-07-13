# Equation of Motion Generator for an \(n\)-Mass Pendulum

Python program that uses SymPy to symbolically derive the equations of motion for a multiple pendulum system composed of \(n\) coupled masses.

The script automatically generates a **Markdown + LaTeX** file ready to be rendered in:

- GitHub Pages
- Quarto
- Jupyter Notebook
- Obsidian
- Typora
- MkDocs
- Any MathJax-compatible blog or website

---

# Features

- Automatic symbolic derivation of differential equations
- Support for an arbitrary number of masses \(n\)
- Automatic `.md` file generation
- LaTeX-aligned equations
- Dot notation for time derivatives:

\[
$\dot{\theta}, \quad \ddot{\theta}$
\]

- Easily extensible for research or educational purposes

---

# Requirements

Install dependencies:

```bash
pip install sympy
```

---

# Usage

Modify the number of masses:

```python
n = 5
```

Run the script:

```bash
python N-MassEqSymboGen.py
```

---

# Output

The program automatically generates a file named:

```text
equations_pendulum_5_masses.md
```

The file contains all differential equations in the following format:

```latex
$$
\begin{aligned}
...
\end{aligned}
$$
```

---

# Example Generated Equation

```latex
$$
\begin{aligned}
& l_{1}^{2} \left(m_{1} + m_{2}\right) \ddot{\theta}_{1}
+ l_{1} l_{2} m_{2} \cos{\left(\theta_{1} - \theta_{2} \right)} \ddot{\theta}_{2}
\\
& \quad + l_{1} l_{2} m_{2} \sin{\left(\theta_{1} - \theta_{2} \right)} \dot{\theta}_{2}^{2}
+ g l_{1} \left(m_{1} + m_{2}\right) \sin{\left(\theta_{1} \right)}
= 0
\end{aligned}
$$
```

---

# Code Structure

## 1. Parameter Definition

The script defines:

- masses \(m_i\)
- lengths \(l_i\)
- gravity \(g\)
- generalized coordinates \(\theta_i(t)\)

---

## 2. Equation Construction

The program computes:

- inertial terms
- angular coupling terms
- centrifugal terms
- gravitational contributions

using symbolic accumulated mass sums.

---

## 3. Markdown Export

The equations are converted into LaTeX using:

```python
vlatex()
```

and automatically written into a `.md` file.

---

# MathJax Compatibility

The generated Markdown file can be directly rendered on MathJax-compatible platforms such as Jupyter, Quarto, Obsidian, Typora, and MkDocs.

# License

MIT License

---

# Citation

If you use this repository, please cite:

Rodríguez Arias, Y. A., & Arellano González, S. "A Unified Structural and Operator Framework for the Linear n-Mass Pendulum"

Stable Zenodo concept DOI: https://doi.org/10.5281/zenodo.20102179
