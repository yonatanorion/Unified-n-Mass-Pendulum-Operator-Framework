"""
O(n) structure-preserving time stepping: Stormer--Verlet on theta'' = -Q theta
with the closed-form tridiagonal Q, compared with the dense transition matrix.

(a) Long-time energy behaviour (n = 3 non-uniform, 10^6 steps).
(b) Second-order convergence against the exact modal solution.
(c) Cost per time step versus n, and one-off cost of precomputing Phi.

Usage:  python Code/legacy/validation/benchmark_symplectic.py [--quick] [--out FIG.png]
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.assembly import assemble_system  # noqa: E402
from src.modal import solve_modal  # noqa: E402
from src.propagator import peano_baker_step  # noqa: E402
from src.structure import dynamic_matrix_tridiagonal  # noqa: E402
from src.symplectic import linear_energy, max_frequency, verlet_propagate  # noqa: E402

M3, L3 = np.array([2.0, 1.5, 0.8]), np.array([1.2, 0.9, 0.6])
TH0 = np.array([0.1, 0.05, 0.02])


def energy_study(n_steps, dt=0.01, every=1000):
    om0 = np.zeros(3)
    E0 = linear_energy(M3, L3, TH0, om0)
    out = {}
    for h in (dt, dt / 2):
        T, TH, OM = verlet_propagate(M3, L3, TH0, om0, h, int(n_steps * dt / h), save_every=int(every * dt / h))
        out[f"Verlet, dt={h:g}"] = (T, np.array([abs(linear_energy(M3, L3, a, b) - E0) / E0 for a, b in zip(TH, OM)]))
    A, _, _ = assemble_system(M3, L3)
    P = peano_baker_step(A, dt)
    R = np.r_[TH0, om0]
    T, E = [0.0], [0.0]
    for s in range(1, n_steps + 1):
        R = P @ R
        if s % every == 0:
            T.append(s * dt)
            E.append(abs(linear_energy(M3, L3, R[:3], R[3:]) - E0) / E0)
    out[f"Transition matrix, dt={dt:g}"] = (np.array(T), np.array(E))
    return out


def convergence_study(t_end=8.0):
    A, M, K = assemble_system(M3, L3)
    ref = solve_modal(M, K, np.r_[TH0, np.zeros(3)], [0.0, t_end])[-1, :3]
    rows = []
    for dt in [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]:
        N = int(round(t_end / dt))
        _, TH, _ = verlet_propagate(M3, L3, TH0, np.zeros(3), dt, N, save_every=N)
        rows.append((dt, np.abs(TH[-1] - ref).max()))
    return rows


def cost_study(quick):
    rng = np.random.default_rng(11)
    ns = [10, 30, 100, 300, 1000, 2000, 3000, 10000, 100000] + ([] if quick else [1000000])
    rows = []
    for n in ns:
        m, l = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n) / n
        d, sub, sup = dynamic_matrix_tridiagonal(m, l)
        th, om = rng.standard_normal(n) * 1e-3, np.zeros(n)
        dt = 0.5 / max_frequency(m, l)
        steps = max(20, int(2e6 / n))
        t0 = time.perf_counter()
        verlet_propagate(m, l, th, om, dt, steps, save_every=steps, check_stability=False)
        t_verlet = (time.perf_counter() - t0) / steps
        t_phi_step = t_phi_pre = np.nan
        if n <= (1000 if quick else 2000):
            A, _, _ = assemble_system(m, l)
            t0 = time.perf_counter(); P = expm(A * dt); t_phi_pre = time.perf_counter() - t0
            R = np.r_[th, om]; k = max(20, int(2e5 / n))
            t0 = time.perf_counter()
            for _ in range(k):
                R = P @ R
            t_phi_step = (time.perf_counter() - t0) / k
        rows.append((n, t_verlet, t_phi_step, t_phi_pre, 8.0 * (2 * n) ** 2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures", "symplectic.png"))
    args = ap.parse_args()

    n_steps = 10**5 if args.quick else 10**6
    en = energy_study(n_steps)
    print("=" * 72)
    print(f"TABLE 3: Relative energy error over {n_steps} steps (n=3 non-uniform)")
    print("=" * 72)
    for k, (T, E) in en.items():
        q = len(E) // 10
        print(f"{k:<28} max {E.max():.2e}  first 10% max {E[:q].max():.2e}  last 10% max {E[-q:].max():.2e}")

    conv = convergence_study()
    print("\nConvergence of Verlet at t = 8 s (max |theta - theta_modal|):")
    for dt, e in conv:
        print(f"  dt = {dt:<7g} error = {e:.3e}")
    p = np.polyfit(np.log([r[0] for r in conv]), np.log([r[1] for r in conv]), 1)[0]
    print(f"  observed order: {p:.2f}")

    cost = cost_study(args.quick)
    print("\n" + "=" * 86)
    print("Cost per step (s): Verlet O(n) vs dense transition matrix; Phi precompute and storage")
    print("=" * 86)
    print(f"{'n':>8} {'Verlet/step':>13} {'Phi/step':>13} {'Phi precompute':>15} {'Phi storage':>13}")
    for n, tv, tp, tpre, mem in cost:
        f = lambda x: f"{x:>13.2e}" if np.isfinite(x) else f"{'--':>13}"
        print(f"{n:>8d} {f(tv)} {f(tp)} {f(tpre):>15} {mem/1e6:>10.1f} MB")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for (k, (T, E)), c in zip(en.items(), ["#1F5E8C", "#6AA5CF", "#B5462F"]):
        ax[0].semilogy(T[1:], E[1:], lw=0.8, color=c, label=k)
    ax[0].set_xlabel("$t$ (s)"); ax[0].set_ylabel("relative energy error"); ax[0].set_title("(a) Long-time energy, $n=3$")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(frameon=False, fontsize=8)
    dts = np.array([r[0] for r in conv]); errs = np.array([r[1] for r in conv])
    ax[1].loglog(dts, errs, "o-", color="#1F5E8C", label="Störmer–Verlet")
    ax[1].loglog(dts, errs[-1] * (dts / dts[-1]) ** 2, "k--", lw=0.8, label="$\\propto \\Delta t^2$")
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("$\\Delta t$ (s)"); ax[1].set_ylabel("max error at $t=8$ s (rad)"); ax[1].set_title("(b) Convergence vs exact modal solution")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(frameon=False)
    n = np.array([r[0] for r in cost], float)
    tv = np.array([r[1] for r in cost]); tp = np.array([r[2] for r in cost])
    ax[2].loglog(n, tv, "s-", color="#1F5E8C", label="Verlet step, $\\mathcal{O}(n)$")
    ok = np.isfinite(tp)
    ax[2].loglog(n[ok], tp[ok], "o-", color="#B5462F", label="$\\Phi$ step, $\\mathcal{O}(n^2)$")
    ax[2].set_xlabel("$n$"); ax[2].set_ylabel("wall time per step (s)"); ax[2].set_title("(c) Cost per time step")
    ax[2].grid(True, which="both", alpha=0.3); ax[2].legend(frameon=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"\nFigure written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
