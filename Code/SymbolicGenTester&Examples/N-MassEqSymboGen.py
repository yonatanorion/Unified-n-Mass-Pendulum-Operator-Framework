import argparse
import sys
import sympy as sp
from sympy.physics.vector import vlatex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate symbolic equations of motion for an n-mass pendulum."
    )
    parser.add_argument(
        "-n",
        "--num-masses",
        type=int,
        default=None,
        help="Number of masses for the pendulum (e.g. 5).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output Markdown file path. Defaults to 'Output_Equations_pendulum_{n}_masses.md'.",
    )
    return parser.parse_args()


def generate_equations(n: int, output_path: str = None) -> str:
    m = sp.symbols(f"m1:{n+1}")
    l = sp.symbols(f"l1:{n+1}")
    t = sp.Symbol("t")
    theta = [sp.Function(f"theta{i}")(t) for i in range(1, n + 1)]
    g = sp.Symbol("g")

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

        # Inertial / acceleration terms
        for k in range(1, n + 1):
            mu = S[max(d, k)]
            if k == d:
                eq_lhs += mu * (l[d - 1] ** 2) * sp.diff(theta[d - 1], t, 2)
            else:
                eq_lhs += (
                    mu
                    * l[d - 1]
                    * l[k - 1]
                    * sp.diff(theta[k - 1], t, 2)
                    * sp.cos(theta[d - 1] - theta[k - 1])
                )

        # Centrifugal / nonlinear coupling terms
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

        # Gravitational torque term
        eq_lhs += S[d] * g * l[d - 1] * sp.sin(theta[d - 1])
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

    for i, lhs in enumerate(equations, start=1):
        md_content += f"## Equation {i}\n\n"
        terms = sp.Add.make_args(lhs)
        latex_terms = [vlatex(term) for term in terms]
        chunks = []
        for j in range(0, len(latex_terms), 2):
            chunk = " + ".join(latex_terms[j : j + 2])
            chunk = chunk.replace("+ -", "- ")
            chunks.append(chunk)

        formatted_eq = " \\\\ \n& \\quad + ".join(chunks)
        formatted_eq = formatted_eq.replace("+ -", "- ")
        formatted_eq = formatted_eq.replace("& \\quad + -", "& \\quad - ")

        md_content += "$$\n\\begin{aligned}\n"
        md_content += f"& {formatted_eq} = 0\n"
        md_content += "\\end{aligned}\n$$\n\n"
        md_content += "---\n\n"

    if output_path is None:
        output_path = f"Output_Equations_pendulum_{n}_masses.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return output_path, len(md_content)


def main():
    args = parse_args()
    if args.num_masses is not None:
        n = args.num_masses
    else:
        try:
            n = int(input("Number of masses for the pendulum:\n"))
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

    if n <= 0:
        print("Error: Number of masses must be a positive integer.")
        sys.exit(1)

    output_path, size = generate_equations(n, args.output)
    print(f"✅ File successfully generated: '{output_path}' (size: {size} characters).")
    print(f"📄 Contains the {n} equations of motion in Markdown + LaTeX format.")


if __name__ == "__main__":
    main()
