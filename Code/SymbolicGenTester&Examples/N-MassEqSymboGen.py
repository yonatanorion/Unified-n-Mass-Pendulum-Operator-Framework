import sympy as sp
from sympy.physics.vector import vlatex

# =======================================================
# Parameters of the n-mass pendulum
# =======================================================
n = int(input("Number of masses for the pendulum:\n"))   # Adjust this value as needed

m = sp.symbols(f'm1:{n+1}')
l = sp.symbols(f'l1:{n+1}')
t = sp.Symbol('t')

theta = [sp.Function(f'theta{i}')(t) for i in range(1, n+1)]

g = sp.Symbol('g')

# =======================================================
# Cumulative mass sums
# =======================================================
S = [0] * (n + 1)

for d in range(1, n + 1):
    S[d] = sum(m[i - 1] for i in range(d, n + 1))

# =======================================================
# Construction of the equations of motion
# =======================================================
equations = []

for d in range(1, n + 1):

    eq_lhs = 0

    # ---------------------------------------------------
    # Inertial / acceleration terms
    # Multiplied by l[d-1] for torque dimensionality
    # ---------------------------------------------------
    for k in range(1, n + 1):

        mu = S[max(d, k)]

        if k == d:

            eq_lhs += (
                mu
                * (l[d - 1] ** 2)
                * sp.diff(theta[d - 1], t, 2)
            )

        else:

            eq_lhs += (
                mu
                * l[d - 1]
                * l[k - 1]
                * sp.diff(theta[k - 1], t, 2)
                * sp.cos(theta[d - 1] - theta[k - 1])
            )

    # ---------------------------------------------------
    # Centrifugal / nonlinear coupling terms
    # The sign is fully determined by sin(theta_d-theta_k).
    # No additional d/k-dependent sign factor is required.
    # ---------------------------------------------------
    for k in range(1, n + 1):

        if k != d:

            mu = S[max(d, k)]

            eq_lhs += (
                mu
                * l[d - 1]
                * l[k - 1]
                * (sp.diff(theta[k - 1], t) ** 2)
                * sp.sin(theta[d - 1] - theta[k - 1])
            )

    # ---------------------------------------------------
    # Gravitational torque term
    # ---------------------------------------------------
    eq_lhs += (
        S[d]
        * g
        * l[d - 1]
        * sp.sin(theta[d - 1])
    )

    equations.append(eq_lhs)

# =======================================================
# Markdown file generation
# =======================================================
md_content = f"# Equations of Motion: {n}-Mass Pendulum\n\n"

md_content += (
    "This document contains the second-order differential "
    "equations obtained for a multiple pendulum system. "
)

md_content += (
    "Dot notation is used for time derivatives, and "
    "multi-line formatting has been applied to improve readability.\n\n"
)

# =======================================================
# Convert equations into LaTeX blocks
# =======================================================
for i, lhs in enumerate(equations, start=1):

    md_content += f"## Equation {i}\n\n"

    # Separate equation terms
    terms = sp.Add.make_args(lhs)

    # Convert each term into LaTeX
    latex_terms = [vlatex(term) for term in terms]

    # Group terms two-by-two for line breaks
    chunks = []

    for j in range(0, len(latex_terms), 2):

        chunk = " + ".join(latex_terms[j:j + 2])

        # Clean sign formatting
        chunk = chunk.replace("+ -", "- ")

        chunks.append(chunk)

    # Build aligned LaTeX equation block
    formatted_eq = " \\\\ \n& \\quad + ".join(chunks)

    formatted_eq = formatted_eq.replace("+ -", "- ")

    formatted_eq = formatted_eq.replace(
        "& \\quad + -",
        "& \\quad - "
    )

    md_content += "$$\n\\begin{aligned}\n"

    md_content += f"& {formatted_eq} = 0\n"

    md_content += "\\end{aligned}\n$$\n\n"

    md_content += "---\n\n"

# =======================================================
# Save Markdown file with dynamic filename
# =======================================================
filename = f"Output_Equations_pendulum_{n}_masses.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(md_content)

# =======================================================
# Console output
# =======================================================
print(
    f"✅ File successfully generated: '{filename}' "
    f"(size: {len(md_content)} characters)."
)

print(
    f"📄 Contains the {n} equations of motion "
    f"in Markdown + LaTeX format."
)
