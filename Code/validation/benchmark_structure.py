"""
Scaling and accuracy study for the closed-form tridiagonal structure.

1. Cost of forming the dynamic matrix Q = M^{-1} K:
   dense route (vectorized M, K + LU solve, O(n^3)) vs closed form (O(n)).
2. Cost and accuracy of all natural frequencies:
   dense generalized eigensolver eigh(K, M) vs tridiagonal Jacobi solver.
3. Non-circular validation: uniform chain of total mass m_tot and length L
   converges to the continuous hanging chain, omega_k = (j_{0,k}/2) sqrt(g/L),
   with j_{0,k} the zeros of the Bessel function J_0.

Usage:  python Code/validation/benchmark_structure.py [--quick] [--out FIG.png]
"""

import argparse
import os
import platform
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import jn_zeros

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.structure import (  # noqa: E402
    apply_dynamic_matrix,
    dynamic_matrix_tridiagonal,
    natural_frequencies,
    suffix_masses,
)

G = 9.81


def dense_mass_stiffness(m, l, g=G):
    """Vectorized dense assembly (fair O(n^2) baseline, no Python loops)."""
    S = suffix_masses(m)
    idx = np.arange(m.size)
    M = S[np.maximum.outer(idx, idx)] * np.outer(l, l)
    K = np.diag(g * S * l)
    return M, K


def best_time(fn, repeat):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def random_chain(rng, n):
    return rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n) / n


def scaling_study(quick):
    rng = np.random.default_rng(2026)
    n_dense = [10, 30, 100, 300, 1000] + ([] if quick else [2000, 3000])
    n_tri = n_dense + ([3000, 10000] if quick else [10000, 30000, 100000])
    rows = []
    for n in sorted(set(n_tri)):
        m, l = random_chain(rng, n)
        rep = 5 if n <= 1000 else 2
        t_q_tri = best_time(lambda: dynamic_matrix_tridiagonal(m, l), rep)
        t_w_tri = best_time(lambda: natural_frequencies(m, l), rep) if n <= 30000 else np.nan
        t_q_den = t_w_den = res_tri = res_den = np.nan
        if n in n_dense:
            M, K = dense_mass_stiffness(m, l)
            t_q_den = best_time(lambda: np.linalg.solve(M, K), rep)
            t_w_den = best_time(lambda: eigh(K, M, eigvals_only=True), rep)
            # Accuracy: residual of each eigenpair against the exact O(n) operator Q.
            from src.structure import modal_matrix
            lam_d, U_d = eigh(K, M)
            omega_t, U_t = modal_matrix(m, l)
            def rel_res(lam, U):
                r = [np.linalg.norm(apply_dynamic_matrix(m, l, U[:, i]) - lam[i] * U[:, i])
                     / (abs(lam[i]) * np.linalg.norm(U[:, i])) for i in range(n)]
                return max(r)
            res_den = rel_res(lam_d, U_d)
            res_tri = rel_res(omega_t**2, U_t)
        rows.append((n, t_q_den, t_q_tri, t_w_den, t_w_tri, res_den, res_tri))
    return rows


def continuum_study():
    L, m_tot, k = 1.0, 1.0, 3
    exact = jn_zeros(0, k) / 2.0 * np.sqrt(G / L)
    rows = []
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        omega = natural_frequencies(np.full(n, m_tot / n), np.full(n, L / n),
                                    select="i", select_range=(0, k - 1))
        rows.append((n, *(np.abs(omega - exact) / exact)))
    return exact, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures", "structure_scaling.png"))
    args = ap.parse_args()

    print(f"Platform: {platform.platform()} | {platform.processor() or platform.machine()} | numpy {np.__version__}")
    rows = scaling_study(args.quick)
    print("\n" + "=" * 110)
    print("TABLE 2: Cost (s) and accuracy of the dense vs closed-form tridiagonal routes")
    print("=" * 110)
    print(f"{'n':>7} {'Q dense':>11} {'Q closed':>11} {'freq dense':>11} {'freq tri':>11} {'resid dense':>12} {'resid tri':>12}")
    for r in rows:
        print(f"{r[0]:>7d} " + " ".join(f"{x:>11.2e}" if np.isfinite(x) else f"{'--':>11}" for x in r[1:5])
              + " " + " ".join(f"{x:>12.1e}" if np.isfinite(x) else f"{'--':>12}" for x in r[5:]))

    exact, crow = continuum_study()
    print("\n" + "=" * 70)
    print("CONTINUUM CHECK (Fig. 3c): Relative error of the three lowest frequencies vs the hanging chain")
    print("         omega_k = (j_{0,k}/2) sqrt(g/L) = " + ", ".join(f"{w:.6f}" for w in exact) + " rad/s")
    print("=" * 70)
    for r in crow:
        print(f"{r[0]:>6d}  " + "  ".join(f"{x:.3e}" for x in r[1:]))
    n_arr = np.array([r[0] for r in crow], float)
    e1 = np.array([r[1] for r in crow])
    p = np.polyfit(np.log(n_arr[-6:]), np.log(e1[-6:]), 1)[0]
    print(f"Observed convergence order (mode 1, last 6 points): {-p:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    n = np.array([r[0] for r in rows], float)
    col = {"dense": "#B5462F", "tri": "#1F5E8C"}
    for j, (title, i_d, i_t) in enumerate([("(a) Forming $Q=M^{-1}K$", 1, 2), ("(b) All natural frequencies", 3, 4)]):
        td = np.array([r[i_d] for r in rows]); tt = np.array([r[i_t] for r in rows])
        ax[j].loglog(n[np.isfinite(td)], td[np.isfinite(td)], "o-", color=col["dense"], label="dense (solve / eigh(K,M))")
        ax[j].loglog(n[np.isfinite(tt)], tt[np.isfinite(tt)], "s-", color=col["tri"], label="closed-form tridiagonal")
        ax[j].set_xlabel("$n$"); ax[j].set_ylabel("wall time (s)"); ax[j].set_title(title)
        ax[j].grid(True, which="both", alpha=0.3); ax[j].legend(frameon=False)
    for k in range(3):
        ax[2].loglog(n_arr, [r[1 + k] for r in crow], "o-", label=f"mode {k+1}")
    ax[2].loglog(n_arr, e1[0] * (n_arr / n_arr[0]) ** -1.0, "k--", lw=0.8, label="$\\propto n^{-1}$")
    ax[2].set_xlabel("$n$"); ax[2].set_ylabel("relative error"); ax[2].set_title("(c) Convergence to hanging chain")
    ax[2].grid(True, which="both", alpha=0.3); ax[2].legend(frameon=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"\nFigure written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
